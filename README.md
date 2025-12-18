# SadTalker Core

Minimal, cleaned version of the SadTalker project focused on programmatic and CLI usage, without the original Gradio / WebUI and demo scaffolding.

## What this repo contains

- Core model code under src/ (preprocess, audio2coeff, face renderer, utilities).
- A CLI entry script inference.py to generate talking-head videos from a single image and an audio file.
- Model files under models/ (models/checkpoints/ and models/gfpgan/weights/ downloaded by setup.py).

Optional helper folders (examples, docs, results) can be ignored or deleted in downstream projects; they are not required by the core code.

## Installation (local project)

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows

# Install PyTorch (version combo known to work with basicsr/gfpgan)
# CPU-only example for macOS/Linux:
pip install "torch==1.12.1" "torchvision==0.13.1" "torchaudio==0.12.1" \
  --extra-index-url https://download.pytorch.org/whl/cpu

# Then install Python dependencies
pip install -r requirements.txt
```

This installs the non-PyTorch dependencies listed in requirements.txt.
PyTorch itself is not in requirements.txt because its wheels are
hardware/OS specific; if you target a different platform or want GPU
builds, use the selector on https://pytorch.org to adjust the command.

You can also use the helper script to install requirements and download all
SadTalker checkpoints and GFPGAN weights in one go:

```bash
python setup.py
```

Or run only specific steps:

```bash
python setup.py --requirements-only  # just pip install -r requirements.txt
python setup.py --models-only        # just download model files
python setup.py --verify             # check that files exist
```

## Basic CLI usage

```bash
python inference.py \
  --driven_audio /path/to/audio.wav \
  --source_image /path/to/image.png \
  --result_dir ./results \
  --size 256 \
  --batch_size 1 \
  --preprocess crop \
  --enhancer gfpgan  # optional, requires GFPGAN weights
```

This writes the output video into the results/ directory.

### Interactive helper (presets)

For a simple, menu-driven CLI with common presets, you can run:

```bash
python cli.py
```

It will ask for the image path, audio path, and whether you want a fast crop preset or a higher-quality full-image preset.

The helper internally builds the correct python inference.py command for you and
stores each run under a unique subdirectory in results/.

## Integrating into another project

For a Flask / chatbot or other backend service:

- Treat this repo as a local package: pip install -e . inside your main project venv.
- Mirror the logic in inference.py inside your own service layer, or wrap it in a small helper that returns the generated video path.
- Ensure the models/checkpoints/ directory is available in your deployed environment and points to the correct model weights.

At runtime you generally:

1. Save the user image and audio to disk.
2. Call the inference script (directly, via cli.py, or via a thin Python wrapper) with those paths and a result_dir you control.
3. Return the resulting mp4 path or URL from your Flask API so your chatbot can display or link it.

If you install this repo into another project with pip install -e ., you can
import the modules directly (e.g. import src.utils.preprocess) and build a
generate_talking_video(...) helper in that project that mirrors what
inference.py does.

### Windows notes

On Windows, inside a venv, you can run:

```powershell
python -m venv .venv
.\.venv\Scripts\activate

# Install PyTorch for your CPU/GPU from pytorch.org
pip install "torch==1.12.1" "torchvision==0.13.1" "torchaudio==0.12.1" `
  --extra-index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
python setup.py --models-only
python cli.py
```

If you see compiler errors when installing basicsr or similar packages, install Build Tools for Visual Studio first so you have a working C/C++ toolchain.

### Troubleshooting

- If basicsr or gfpgan fail to install with wheel warnings, upgrade pip/setuptools/wheel in your venv and retry.
- If you see ModuleNotFoundError for torchvision.transforms.functional_tensor, pin torch/torchvision to 1.12.1 / 0.13.1 as in the README install example.
- On Apple Silicon (M1/M2), use a PyTorch build with MPS support so the heavy steps can run on the GPU instead of CPU.

### Cleanup notes

- The scripts/ folder (download_models.sh and the old WebUI extension) is not
  used by the current CLI/core flow; you can delete it if you like.
- examples/, docs/, and results/ are optional convenience folders; they are not
  required for inference and can be removed in a downstream project.
