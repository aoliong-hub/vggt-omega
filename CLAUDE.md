# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGT-Omega is a feed-forward model for camera pose and depth estimation from multiple images. Given a set of images (or video frames), it predicts per-frame camera extrinsics/intrinsics and dense depth maps in a single forward pass. It originates from Meta AI / University of Oxford (FAIR Noncommercial Research License).

Two checkpoints exist: `VGGT-Omega-1B-512` (resolution 512, no text alignment) and `VGGT-Omega-1B-256-Text-Alignment` (resolution 256, text-aligned). Checkpoints require access approval on HuggingFace (`facebook/VGGT-Omega`).

## Setup

```bash
pip install -r requirements.txt   # torch 2.3, torchvision 0.18, numpy<2, einops, etc.
pip install -e .                   # editable install of vggt_omega package
pip install -r requirements_demo.txt  # for Gradio demo (gradio, trimesh, scipy, onnxruntime)
```

Requires Python >= 3.10 and CUDA.

## Running

**Inference script:**
```bash
python demo.py  # edit checkpoint_path and image_names inside the file
```

**Gradio demo:**
```bash
python demo_gradio.py --checkpoint path/to/model.pt --image-resolution 512
```
Accepts `--server-name`, `--server-port` (default 8001), `--share`. Outputs saved to `demo_outputs/`.

## Architecture

The model (`VGGTOmega` in `vggt_omega/models/vggt_omega.py`) has three components:

1. **Aggregator** (`vggt_omega/models/aggregator.py`): A DINOv3-based ViT patch embedder followed by 24 alternating self-attention blocks. Each layer runs a per-frame attention block then an inter-frame attention block. Inter-frame attention alternates between "global" (all tokens across all frames) and "register" (only camera+register tokens across frames). Outputs cached at layers [4, 11, 17, 23] for multi-scale feature extraction.

2. **Prediction Heads** (`vggt_omega/models/heads/`):
   - `CameraHead`: Extracts camera+register tokens, runs 4 transformer blocks, predicts 9D pose encoding per frame (translation 3D + quaternion 4D + vertical/horizontal FoV 2D).
   - `DenseHead`: DPT-style multi-scale fusion over cached layers. Predicts per-pixel depth (log-space, exp activated) and depth confidence. Processes frames in chunks of 8 for memory efficiency.
   - `TextAlignmentHead`: Optional, enabled with `VGGTOmega(enable_alignment=True)`.

3. **Utilities** (`vggt_omega/utils/`):
   - `load_fn.py`: `load_and_preprocess_images()` — two resize modes: `"balanced"` (preserves token count ≈ resolution²) and `"max_size"` (longest side = resolution). Crops extreme aspect ratios to [0.5, 2.0]. Images with different sizes are padded.
   - `pose_enc.py`: `encoding_to_camera()` decodes 9D pose encoding into 3x4 extrinsic matrices (camera-from-world, OpenCV convention) and 3x3 intrinsic matrices.
   - `rotation.py`: Quaternion ↔ rotation matrix conversions.
   - `geometry.py`: Geometric utilities.

**Token layout**: Each frame's token sequence is `[camera_token (1), register_tokens (16), patch_tokens (H/16 × W/16)]`. The `patch_token_start` index (= 17) separates camera/register tokens from patch tokens throughout the pipeline.

**Precision**: Forward pass uses `torch.autocast` with bfloat16 (or float16 fallback) for the aggregator. Heads run in float32.

## Visualization

`visual_util.py` converts predictions to GLB scenes via trimesh. `predictions_to_glb()` unprojects depth maps to world-space point clouds, filters by confidence percentile, and renders camera frustums. Optional sky segmentation uses a downloaded ONNX model (`skyseg.onnx`).
