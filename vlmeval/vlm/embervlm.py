"""
EmberVLM Model Adapter for VLMEvalKit

This adapter allows EmberVLM to be evaluated using VLMEvalKit's
rule-based evaluation system.

CRITICAL CONSTRAINTS (FREE MODE ONLY):
- NO paid APIs
- NO LLM-as-a-judge evaluation
- Rule-based scoring only (exact match, regex, MCQ)
- Fully offline operation

Supported benchmarks:
- MMStar, MMBench (multiple choice)
- MMMU (multiple choice)
- ScienceQA, AI2D (multiple choice)
- TextVQA, DocVQA (exact match)
- ChartQA (numeric/exact match)

Usage:
    model = EmberVLMEval(model_path="/path/to/checkpoint")
    response = model.generate(message, dataset="MMStar")

Author: EmberVLM Team
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseModel

logger = logging.getLogger(__name__)


class EmberVLMEval(BaseModel):
    """
    EmberVLM adapter for VLMEvalKit evaluation.

    Designed for FREE/RULE-BASED evaluation only:
    - NO padding during generation (prevents CUDA indexing errors)
    - NO KV cache (use_cache=False for stability)
    - NO tokenizer/config mutation
    - Simple, explicit generation
    """

    INSTALL_REQ = False  # No additional installation required
    INTERLEAVE = False   # Does not support interleaved image-text

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        max_new_tokens: int = 512,
        **kwargs,
    ):
        """
        Initialize EmberVLM for VLMEvalKit.

        Args:
            model_path: Path to EmberVLM checkpoint directory
            device: Device to use ('cuda' or 'cpu')
            max_new_tokens: Maximum tokens to generate
        """
        super().__init__()

        # Get model path from environment if not provided
        if model_path is None:
            model_path = os.environ.get("EMBERVLM_CHECKPOINT")
        if model_path is None:
            raise ValueError(
                "model_path must be provided or EMBERVLM_CHECKPOINT env var must be set"
            )

        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        logger.info(f"[EmberVLM] Loading from {model_path}")
        logger.info(f"[EmberVLM] Device: {self.device}")
        logger.info(f"[EmberVLM] FREE MODE: No LLM judges, rule-based only")

        self._load_model()
        self._load_tokenizer()
        self._validate_setup()

        # Generation kwargs - deterministic for evaluation
        self.gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "use_cache": False,  # Disable KV cache for stability
        }

        logger.info(f"[EmberVLM] Loaded successfully, vocab_size={self._vocab_size}")

    def _load_model(self):
        """Load EmberVLM model."""
        try:
            # Add EmberVLM to path if needed
            embervlm_candidates = [
                Path(__file__).resolve().parents[4] / "EmberVLM",
                Path.home() / "EmberVLM",
                Path("/root/EmberVLM"),
            ]

            for candidate in embervlm_candidates:
                if candidate.exists() and str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
                    break

            from embervlm.models import EmberVLM as EmberVLMModel

            self.model = EmberVLMModel.from_pretrained(str(self.model_path))
            self.model = self.model.to(self.device).eval()
            self.image_preprocessor = getattr(self.model, "image_preprocessor", None)

        except ImportError as e:
            logger.error(f"[EmberVLM] Import failed: {e}")
            raise
        except Exception as e:
            logger.error(f"[EmberVLM] Model loading failed: {e}")
            raise

    def _load_tokenizer(self):
        """Load tokenizer with fallback."""
        from transformers import AutoTokenizer

        # Try checkpoint tokenizer directory
        tokenizer_path = self.model_path / "tokenizer"
        if not tokenizer_path.exists():
            tokenizer_path = self.model_path.parent.parent / "tokenizer"

        if tokenizer_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            logger.info(f"[EmberVLM] Tokenizer: {tokenizer_path}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
            logger.warning("[EmberVLM] Using fallback tokenizer")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _validate_setup(self):
        """Validate model/tokenizer setup."""
        self._vocab_size = None
        try:
            if hasattr(self.model, "language_model"):
                lm = self.model.language_model
                if hasattr(lm, "model") and hasattr(lm.model, "get_input_embeddings"):
                    emb = lm.model.get_input_embeddings()
                    if emb is not None:
                        self._vocab_size = emb.weight.shape[0]
                elif hasattr(lm, "get_input_embeddings"):
                    emb = lm.get_input_embeddings()
                    if emb is not None:
                        self._vocab_size = emb.weight.shape[0]
        except Exception:
            pass

        if self._vocab_size is None:
            self._vocab_size = len(self.tokenizer)

        # Use smaller to be safe
        tokenizer_size = len(self.tokenizer)
        if tokenizer_size != self._vocab_size:
            logger.warning(f"[EmberVLM] Vocab mismatch: tokenizer={tokenizer_size}, model={self._vocab_size}")
            self._vocab_size = min(self._vocab_size, tokenizer_size)

    def _safe_tokenize(self, text: str) -> torch.Tensor:
        """Tokenize with safety checks - NO PADDING."""
        max_len = 1024
        if hasattr(self.model, "config"):
            cfg = self.model.config
            max_pos = getattr(cfg, "language_max_length", 1024)
            num_visual = getattr(cfg, "num_visual_tokens", 8)
            max_len = min(max_len, max_pos - num_visual - 50)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        )

        input_ids = inputs["input_ids"]

        # Clamp to valid vocab range
        if self._vocab_size > 0:
            oov_mask = (input_ids >= self._vocab_size) | (input_ids < 0)
            if oov_mask.any():
                safe_id = min(self.tokenizer.eos_token_id or 0, self._vocab_size - 1)
                input_ids = input_ids.clone()
                input_ids[oov_mask] = safe_id

        return input_ids.to(self.device)

    def _load_image(self, image_input) -> Optional[torch.Tensor]:
        """Load and preprocess image."""
        try:
            if isinstance(image_input, str):
                img = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                img = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                img = Image.fromarray(image_input).convert("RGB")
            else:
                return None

            if self.image_preprocessor is None:
                return None

            img_np = np.array(img).astype("float32") / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device, dtype=torch.float32)
            pixel_values = self.image_preprocessor(img_tensor)
            return pixel_values.to(self.device)

        except Exception as e:
            logger.error(f"[EmberVLM] Image loading failed: {e}")
            return None

    def use_custom_prompt(self, dataset: str) -> bool:
        """Check if custom prompt should be used."""
        return False

    def generate_inner(self, message: List[Dict], dataset: str = None) -> str:
        """
        Generate response for VLMEvalKit input.

        Args:
            message: List of dicts with 'type' and 'value'
            dataset: Dataset name for logging

        Returns:
            Generated text
        """
        # Extract text and image
        text_parts = []
        image_input = None

        for item in message:
            item_type = item.get("type", "")
            item_value = item.get("value", "")

            if item_type == "text":
                text_parts.append(str(item_value))
            elif item_type == "image":
                image_input = item_value

        # Build prompt
        prompt = "\n".join(text_parts)
        prompt = prompt.replace("<|image|>", "").replace("<image>", "")
        prompt = prompt.replace("<img>", "").replace("</img>", "").strip()

        if not prompt:
            prompt = "Describe what you see in the image."

        # Tokenize
        input_ids = self._safe_tokenize(prompt)

        # Process image
        pixel_values = None
        image_positions = None
        if image_input is not None:
            pixel_values = self._load_image(image_input)
            if pixel_values is not None:
                image_positions = torch.zeros(1, dtype=torch.long, device=self.device)

        # Generate
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=None,
                    image_positions=image_positions,
                    **self.gen_kwargs,
                )

                if isinstance(outputs, torch.Tensor):
                    outputs = torch.clamp(outputs, 0, self._vocab_size - 1)
                    prompt_len = input_ids.size(1)
                    generated = self.tokenizer.decode(
                        outputs[0][prompt_len:], skip_special_tokens=True
                    )
                else:
                    generated = str(outputs)

                return generated.strip()

            except RuntimeError as e:
                if "CUDA" in str(e) or "assert" in str(e).lower():
                    logger.error(f"[EmberVLM] CUDA error: {e}")
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    except:
                        pass
                    return ""
                raise

