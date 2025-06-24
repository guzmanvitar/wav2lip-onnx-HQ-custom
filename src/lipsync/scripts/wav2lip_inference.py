#!/usr/bin/env python3
"""
Wav2Lip ONNX Inference CLI Entry Point
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if __name__ == "__main__":
    from lipsync.models.wav2lip.inference import main

    main()
