import io
import socket
import time
from threading import Lock
from typing import Optional

import requests
import urllib3.util.connection as urllib3_cn
from PIL import Image

from app.config import settings
from app.core.exceptions import CaptionGenerationException, ModelNotReadyException
from app.core.logging_config import get_logger

logger = get_logger(__name__)


# Fix for a known Docker/hosting-provider networking issue: some containers
# (including Render's free tier) report having an IPv6 network interface but
# have no actual IPv6 route to the internet. Python's DNS resolution then
# tries an IPv6 lookup first, gets no usable address, and fails with
# "[Errno -5] No address associated with hostname" — even though a normal
# IPv4 lookup would have worked fine. Forcing urllib3 (used by `requests`) to
# only ever request IPv4 addresses avoids this entirely.
def _allowed_gai_family():
    return socket.AF_INET


urllib3_cn.allowed_gai_family = _allowed_gai_family


class BlipCaptionService:
    """Singleton wrapper that calls the Hugging Face Inference API for BLIP image
    captioning, instead of loading the model locally. This keeps RAM usage low
    enough to run on free hosting tiers (e.g. Render's 512MB free plan), since the
    actual model runs on Hugging Face's servers, not on our own."""

    _instance: Optional["BlipCaptionService"] = None
    _lock = Lock()

    def __new__(cls) -> "BlipCaptionService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.is_ready = False
        # Kept for compatibility with code (like the health check endpoint) that
        # reads blip_service.device. There is no local device anymore since the
        # model runs on Hugging Face's servers, so this is just a descriptive label.
        self.device = "huggingface-inference-api"
        self._initialized = True

    def load_model(self) -> None:
        """No local model to load anymore. We just confirm an API token is configured."""
        if self.is_ready:
            return

        if not getattr(settings, "HF_API_TOKEN", None):
            logger.error(
                "HF_API_TOKEN is not set — cannot call the Hugging Face Inference API"
            )
            self.is_ready = False
            return

        self.is_ready = True
        logger.info(
            f"BLIP captioning will use the Hugging Face Inference API for model "
            f"'{settings.MODEL_NAME}' (no local model load, low RAM usage)"
        )

    def unload_model(self) -> None:
        self.is_ready = False

    def _call_hf_api_with_retry(
        self, api_url: str, headers: dict, image_bytes: bytes, max_network_retries: int = 3
    ) -> requests.Response:
        """Calls the Hugging Face Inference API, retrying a few times if a
        network-level error occurs (e.g. a transient DNS resolution failure on
        the hosting provider's container). This is separate from the "model is
        warming up" retry loop in generate_caption, which handles HTTP 503s."""
        last_network_error: Optional[Exception] = None

        for attempt in range(max_network_retries):
            try:
                return requests.post(
                    api_url, headers=headers, data=image_bytes, timeout=60
                )
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as exc:
                last_network_error = exc
                wait_time = 2 * (attempt + 1)
                logger.warning(
                    f"Network error calling Hugging Face API (attempt "
                    f"{attempt + 1}/{max_network_retries}): {exc}. "
                    f"Retrying in {wait_time}s..."
                )
                if attempt < max_network_retries - 1:
                    time.sleep(wait_time)

        raise CaptionGenerationException(
            f"Could not reach Hugging Face API after {max_network_retries} attempts: "
            f"{last_network_error}"
        )

    def generate_caption(
        self,
        image: Image.Image,
        max_length: int = None,
        num_beams: int = None,
    ) -> tuple[str, float, int]:
        if not self.is_ready:
            raise ModelNotReadyException()

        start_time = time.time()

        try:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()

            api_url = f"https://api-inference.huggingface.co/models/{settings.MODEL_NAME}"
            headers = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}

            caption = None

            # Hugging Face's free API can take a few seconds to "wake up" the model
            # on the first request (cold start). Retry a few times if it's still loading.
            for attempt in range(5):
                response = self._call_hf_api_with_retry(api_url, headers, image_bytes)

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and result and "generated_text" in result[0]:
                        caption = result[0]["generated_text"].strip()
                        break
                    raise CaptionGenerationException(f"Unexpected response format: {result}")

                if response.status_code == 503:
                    wait_time = 3
                    try:
                        wait_time = float(response.json().get("estimated_time", 3))
                    except Exception:
                        pass
                    logger.info(
                        f"Model is warming up on Hugging Face, waiting {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/5)..."
                    )
                    time.sleep(min(wait_time, 15))
                    continue

                raise CaptionGenerationException(
                    f"Hugging Face API error {response.status_code}: {response.text}"
                )

            if caption is None:
                raise CaptionGenerationException(
                    "Model did not respond in time, please try again"
                )

            # The Hugging Face Inference API does not return a confidence score,
            # so we report a fixed reasonable default instead.
            confidence = 0.85

        except CaptionGenerationException:
            raise
        except Exception as exc:
            logger.error(f"Caption generation failed: {exc}")
            raise CaptionGenerationException(str(exc))

        processing_time_ms = int((time.time() - start_time) * 1000)
        return caption, confidence, processing_time_ms


blip_service = BlipCaptionService()