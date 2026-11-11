
import base64
import os
import re
import threading
import time
from datetime import datetime, timezone

import requests
from flask import Blueprint, jsonify, request

IMAGE_API_KEY = os.getenv("IMAGE_API_KEY")
IMAGE_API_URL = os.getenv(
    "IMAGE_API_URL",
    "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
)
DAILY_LIMIT = int(os.getenv("IMAGE_DAILY_LIMIT", "5"))
GLOBAL_LIMIT = int(os.getenv("IMAGE_GLOBAL_LIMIT", "100"))
PROMPT_MAX = int(os.getenv("IMAGE_PROMPT_MAX", "200"))
PROMPT_MIN = 3
REQUEST_TIMEOUT = 60
MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Added to every prompt so results stay classroom-friendly.
STYLE_PREFIX = "Simple, clear, child-friendly educational illustration, no text or letters: "

# Words we refuse outright. This is a basic filter, not a guarantee.
BLOCKED_WORDS = {
    "nude", "naked", "nsfw", "porn", "sex", "sexual", "gore", "blood", "kill", "murder",
    "suicide", "weapon", "gun", "bomb", "terror", "drug", "cocaine", "heroin",
}


class ImageError(RuntimeError):
    """Image generation failed; the message is safe to show the user."""


class LimitReached(ImageError):
    """A usage limit was hit."""



class UsageLimiter:
    """Counts images per visitor and for the whole site, per UTC day.

    Stored in memory, so counts reset when the server restarts and are not shared
    between several gunicorn workers. For strict limits across restarts, replace
    this class with one that keeps the counts in Supabase or Redis (same methods).
    """

    def __init__(self, per_visitor, overall):
        self.per_visitor = per_visitor
        self.overall = overall
        self._lock = threading.Lock()
        self._day = self._today()
        self._visitors = {}
        self._total = 0

    @staticmethod
    def _today():
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _roll_day(self):
        if self._today() != self._day:
            self._day, self._visitors, self._total = self._today(), {}, 0

    def remaining(self, visitor):
        with self._lock:
            self._roll_day()
            return max(0, self.per_visitor - self._visitors.get(visitor, 0))

    def reserve(self, visitor):
        """Use up one image for this visitor, or raise LimitReached."""
        with self._lock:
            self._roll_day()
            if self._total >= self.overall:
                raise LimitReached("Image creation is paused for today. Please try again tomorrow.")
            if self._visitors.get(visitor, 0) >= self.per_visitor:
                raise LimitReached(f"You have used your {self.per_visitor} images for today. Come back tomorrow!")
            self._visitors[visitor] = self._visitors.get(visitor, 0) + 1
            self._total += 1

    def refund(self, visitor):
        """Give the image back if generation failed."""
        with self._lock:
            if self._visitors.get(visitor, 0) > 0:
                self._visitors[visitor] -= 1
                self._total = max(0, self._total - 1)


limiter = UsageLimiter(DAILY_LIMIT, GLOBAL_LIMIT)


# --------------------------------------------------------------------------
# Prompt checking
# --------------------------------------------------------------------------
def clean_prompt(prompt):
    """Validate the prompt and return the text that will be sent to the provider."""
    prompt = re.sub(r"\s+", " ", (prompt or "")).strip()
    if len(prompt) < PROMPT_MIN:
        raise ValueError("Please describe the picture you want (for example: a water cycle diagram).")
    if len(prompt) > PROMPT_MAX:
        raise ValueError(f"Description is too long. Please keep it under {PROMPT_MAX} characters.")
    words = set(re.findall(r"[a-z]+", prompt.lower()))
    if words & BLOCKED_WORDS:
        raise ValueError("That description is not suitable for an educational picture.")
    return STYLE_PREFIX + prompt



def generate_image_bytes(full_prompt):
    """Call the image provider and return (image_bytes, mime_type)."""
    if not IMAGE_API_KEY:
        raise ImageError("Image creation is not switched on for this site.")

    try:
        response = requests.post(
            IMAGE_API_URL,
            headers={"Authorization": f"Bearer {IMAGE_API_KEY}"},
            json={"inputs": full_prompt},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.RequestException as e:
        raise ImageError(f"Could not reach the image service: {e}")

    if response.status_code in (401, 403):
        raise ImageError("Image service rejected the key. Check IMAGE_API_KEY.")
    if response.status_code == 503:
        raise ImageError("The image service is starting up. Please try again in a minute.")
    if response.status_code == 429:
        raise ImageError("The image service is busy. Please try again shortly.")
    if response.status_code != 200:
        raise ImageError(f"Image service error (status {response.status_code}).")

    mime = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        raise ImageError("The image service did not return a picture.")
    if len(response.content) > MAX_IMAGE_BYTES:
        raise ImageError("The picture was too large to send.")
    return response.content, mime



image_bp = Blueprint("image_service", __name__)


def visitor_id():
    # Behind Render's proxy, add this to chatbot.py so remote_addr is the real visitor:
    #   from werkzeug.middleware.proxy_fix import ProxyFix
    #   app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)
    return request.remote_addr or "unknown"


@image_bp.route("/image-status", methods=["GET"])
def image_status():
    return jsonify({
        "enabled": bool(IMAGE_API_KEY),
        "remaining_today": limiter.remaining(visitor_id()) if IMAGE_API_KEY else 0,
        "daily_limit": DAILY_LIMIT,
    })


@image_bp.route("/generate-image", methods=["POST"])
def generate_image_route():
    if not IMAGE_API_KEY:
        return jsonify({"error": "Image creation is not switched on for this site."}), 404

    data = request.get_json(silent=True) or {}
    visitor = visitor_id()
    try:
        full_prompt = clean_prompt(data.get("prompt", ""))
        limiter.reserve(visitor)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except LimitReached as e:
        return jsonify({"error": str(e), "remaining_today": 0}), 429

    try:
        image, mime = generate_image_bytes(full_prompt)
    except ImageError as e:
        limiter.refund(visitor)          # a failed attempt does not count
        return jsonify({"error": str(e)}), 502

    encoded = base64.b64encode(image).decode("ascii")
    return jsonify({
        "image": f"data:{mime};base64,{encoded}",
        "remaining_today": limiter.remaining(visitor),
    })