# Wav2Lip ONNX HQ Custom

A modular lip-sync toolkit that generates lip movements from audio input. This repository is a **custom fork** of the excellent [instant-high/wav2lip-onnx-HQ](https://github.com/instant-high/wav2lip-onnx-HQ) project, featuring a complete codebase refactor for better maintainability and extensibility.

## 🎯 What This Project Does

This toolkit performs **audio-driven lip synchronization** - it takes a video (or image) containing a face and an audio file, then generates a new video where the person's lips move in sync with the audio.

## ✨ Key Features

### 🎭 **Advanced Face Processing**
- **Multi-face detection** with automatic target face selection
- **Face alignment** supporting ±60° head tilt
- **Face recognition** for specific person targeting
- **Face masking** with multiple techniques (static, blendmasker, x-seg occlusion)

### 🎨 **Enhancement Options**
- **4 different face enhancers**: GFPGAN, CodeFormer, RestoreFormer, GPEN
- **Adjustable enhancement blending** (1-10 levels)
- **Frame enhancement** with RealESRGAN
- **Audio denoising** to reduce unwanted lip movements

### 🎬 **Video Processing**
- **Static image support** (creates video from single photo)
- **Pingpong looping** for seamless loops
- **Cut-in/cut-out** frame selection
- **Fade in/out effects**
- **Multiple face modes** for different face shapes
- **High-quality output** options

### ⚡ **Performance**
- **ONNX-optimized** for fast inference
- **CPU and GPU support** (CUDA compatible)
- **Memory management** with garbage collection
- **Modular architecture** for easy extension

## 🏗️ Project Structure

```
src/lipsync/
├── core/                    # Shared functionality
│   ├── audio/              # Audio processing (mel spectrograms, denoising)
│   ├── face/               # Face detection, recognition, alignment
│   ├── enhancement/        # Face and frame enhancement models
│   └── segmentation/       # Face masking and segmentation
├── models/                 # Model-specific implementations
│   └── wav2lip/           # Wav2Lip ONNX models and inference
└── scripts/               # Executable entry points
    └── wav2lip_inference.py
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/wav2lip-onnx-HQ-custom.git
   cd wav2lip-onnx-HQ-custom
   ```

2. **Set up the environment:**
   ```bash
   uv install python 3.11.4
   uv sync
   ```

### Basic Usage

```bash
python src/lipsync/scripts/wav2lip_inference.py \
    --checkpoint_path src/lipsync/models/wav2lip/models/wav2lip.onnx \
    --face input_video.mp4 \
    --audio input_audio.wav \
    --outfile output_result.mp4
```

## 📋 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoint_path` | Path to Wav2Lip ONNX model | Required |
| `--face` | Input video/image with face | Required |
| `--audio` | Input audio file | Required |
| `--outfile` | Output video path | `results/result_voice.mp4` |
| `--enhancer` | Face enhancer: none/gfpgan/codeformer/restoreformer/gpen | `none` |
| `--blending` | Enhancement blending level (1-10) | `10` |
| `--face_mask` | Use face masking | `False` |
| `--denoise` | Denoise audio to reduce unwanted movements | `False` |
| `--face_mode` | Face crop mode: 0=portrait, 1=square | `0` |
| `--resize_factor` | Reduce resolution by this factor | `1` |
| `--static` | Use only first frame for inference | `False` |
| `--pingpong` | Create pingpong loop | `False` |

For a complete list of options, run:
```bash
python src/lipsync/scripts/wav2lip_inference.py --help
```

## 🔧 Technical Details

### Model Architecture
This project uses ONNX-converted models for optimal performance:
- **Wav2Lip**: Core lip-sync generation
- **RetinaFace**: Face detection and alignment
- **Face Recognition**: Target face selection
- **BlendMasker/X-Seg**: Face masking and segmentation
- **GFPGAN/CodeFormer/etc.**: Face enhancement
- **Resemble Denoiser**: Audio denoising

### Performance Optimization
- **ONNX Runtime**: Fast inference with CPU/GPU support
- **Memory Management**: Automatic garbage collection
- **Batch Processing**: Efficient frame processing
- **Modular Design**: Easy to extend with new models

## 🤝 Credits and Acknowledgments

This project is based on the excellent work of many researchers and developers:

### **Original Repository**
- **[instant-high/wav2lip-onnx-HQ](https://github.com/instant-high/wav2lip-onnx-HQ)** - The base repository this project is forked from

### **Core Technologies**
- **[Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)** - Original Wav2Lip implementation
- **[harisreedhar/Face-Upscalers-ONNX](https://github.com/harisreedhar/Face-Upscalers-ONNX)** - Face enhancement models
- **[neuralchen/SimSwap](https://github.com/neuralchen/SimSwap)** - Face detection and alignment
- **[facefusion/facefusion-assets](https://github.com/facefusion/facefusion-assets/releases)** - Face occlusion models
- **[mapooon/BlendFace](https://github.com/mapooon/BlendFace)** - BlendMasker face masking
- **[jahongir7174/FaceID](https://github.com/jahongir7174/FaceID)** - Face recognition for specific targeting
- **[skeskinen/resemble-denoise-onnx-inference](https://github.com/skeskinen/resemble-denoise-onnx-inference)** - Audio denoising

## 📄 License

This project maintains the same license as the original [instant-high/wav2lip-onnx-HQ](https://github.com/instant-high/wav2lip-onnx-HQ) repository. Please refer to the original repository for licensing information.

## 🤝 Contributing

This is a custom fork focused on code organization and maintainability. For feature requests or bug reports related to the core Wav2Lip functionality, please consider contributing to the [original repository](https://github.com/instant-high/wav2lip-onnx-HQ).
