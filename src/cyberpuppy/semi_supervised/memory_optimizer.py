"""
Memory optimization utilities for semi-supervised training.
Provides GPU memory management and optimization strategies.
"""

import gc
import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Memory optimization manager for training.
    Handles GPU memory cleanup, gradient checkpointing, and mixed precision training.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_gradient_checkpointing: bool = False,
        use_mixed_precision: bool = False,
    ):
        """
        Initialize memory optimizer.

        Args:
            device: Torch device (cuda/cpu)
            use_gradient_checkpointing: Enable gradient checkpointing
            use_mixed_precision: Enable mixed precision training
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self._enabled = str(self.device).startswith("cuda")

        logger.info(
            f"MemoryOptimizer initialized: device={self.device}, "
            f"gradient_checkpointing={use_gradient_checkpointing}, "
            f"mixed_precision={use_mixed_precision}"
        )

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self._enabled:
            gc.collect()
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")

    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply memory optimizations to model.

        Args:
            model: Model to optimize

        Returns:
            Optimized model
        """
        if self.use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        return model

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.

        Returns:
            Dictionary with memory usage information
        """
        if not self._enabled:
            return {"device": str(self.device), "enabled": False}

        memory_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        max_memory_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)

        return {
            "device": str(self.device),
            "enabled": True,
            "memory_allocated_gb": round(memory_allocated, 2),
            "memory_reserved_gb": round(memory_reserved, 2),
            "max_memory_allocated_gb": round(max_memory_allocated, 2),
        }

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if self._enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
            logger.debug("Peak memory stats reset")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.clear_cache()
