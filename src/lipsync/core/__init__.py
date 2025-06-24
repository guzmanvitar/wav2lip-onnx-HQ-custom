"""
Core functionality for lip-sync processing.

This module contains all model-agnostic functionality including:
- Audio processing
- Face detection and alignment
- Face enhancement
- Segmentation
- Configuration (hparams)
"""

from . import audio, enhancement, face, hparams, segmentation

__all__ = ["audio", "enhancement", "face", "segmentation", "hparams"]
