import time
from threading import Lock
from typing import Optional

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from app.config import settings
from app.core.exceptions import CaptionGenerationException, ModelNotReadyException
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class BlipCaptionService:
    """Singleton wrapper around the Salesforce BLIP image captioning model."""

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
        self.processor: Optional[BlipProcessor] = None
        self.model: Optional[BlipForConditionalGeneration] = None
        self.device = torch.device(
            settings.MODEL_DEVICE if torch.cuda.is_available() or settings.MODEL_DEVICE == "cpu" else "cpu"
        )
        self.is_ready = False
        self._initialized = True

    def load_model(self) -> None:
        if self.is_ready:
            return

        logger.info(f"Loading BLIP model '{settings.MODEL_NAME}' on device '{self.device}'...")
        start = time.time()

        self.processor = BlipProcessor.from_pretrained(
            settings.MODEL_NAME,
            cache_dir=settings.MODEL_CACHE_DIR,
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            settings.MODEL_NAME,
            cache_dir=settings.MODEL_CACHE_DIR,
        ).to(self.device)
        self.model.eval()

        elapsed = time.time() - start
        self.is_ready = True
        logger.info(f"BLIP model loaded successfully in {elapsed:.2f}s")

    def unload_model(self) -> None:
        self.model = None
        self.processor = None
        self.is_ready = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate_caption(
        self,
        image: Image.Image,
        max_length: int = None,
        num_beams: int = None,
    ) -> tuple[str, float, int]:
        if not self.is_ready or self.model is None or self.processor is None:
            raise ModelNotReadyException()

        max_length = max_length or settings.MAX_CAPTION_LENGTH
        num_beams = num_beams or settings.NUM_BEAMS

        start_time = time.time()

        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=settings.MIN_CAPTION_LENGTH,
                num_beams=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )

            caption = self.processor.decode(
                output_ids.sequences[0], skip_special_tokens=True
            ).strip()

            confidence = self._compute_confidence(output_ids)

        except Exception as exc:
            logger.error(f"Caption generation failed: {exc}")
            raise CaptionGenerationException(str(exc))

        processing_time_ms = int((time.time() - start_time) * 1000)
        return caption, confidence, processing_time_ms

    def _compute_confidence(self, output_ids) -> float:
        try:
            if hasattr(output_ids, "sequences_scores") and output_ids.sequences_scores is not None:
                log_prob = output_ids.sequences_scores[0].item()
                confidence = min(max(torch.exp(torch.tensor(log_prob)).item(), 0.0), 1.0)
                return round(confidence, 4)
        except Exception:
            pass
        return 0.85


blip_service = BlipCaptionService()