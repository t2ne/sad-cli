#!/usr/bin/env python3
"""SadTalker environment/setup helper

- Installs Python requirements (excluding PyTorch, which you should install from pytorch.org)
- Downloads SadTalker checkpoints and GFPGAN/facexlib weights (replacing scripts/download_models.sh)
- Verifies that required files exist

Usage:
  python setup.py              # full setup (requirements + models + verify)
  python setup.py --requirements-only
  python setup.py --models-only
  python setup.py --verify
"""

import sys
import subprocess
import shutil
from pathlib import Path
from typing import Iterable, Tuple, Optional
import urllib.request


class SadTalkerSetup:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent
        models_root = self.project_root / "models"
        self.checkpoints_dir = models_root / "checkpoints"
        self.gfpgan_weights_dir = models_root / "gfpgan" / "weights"

    # --- helpers ---------------------------------------------------------

    def print_header(self, title: str) -> None:
        bar = "=" * 60
        print(f"\n{bar}\n{title}\n{bar}")

    def run_command(self, cmd: str, description: Optional[str] = None) -> bool:
        if description:
            print(f"{description}...")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout.strip())
            if result.returncode != 0:
                print(f"Command failed: {cmd}")
                if result.stderr:
                    print(result.stderr.strip())
                return False
            return True
        except Exception as exc:  # pragma: no cover - best effort helper
            print(f"Error running command '{cmd}': {exc}")
            return False

    # --- steps -----------------------------------------------------------

    def check_python_version(self) -> bool:
        v = sys.version_info
        print(f"Python version: {v.major}.{v.minor}.{v.micro}")

        if v.major != 3 or v.minor < 8:
            print("SadTalker requires Python 3.8+.")
            return False
        if v.minor > 10:
            print("Warning: Python 3.11+ may have compatibility issues with some deps.")
        return True

    def check_system_tools(self) -> None:
        """Check for required system tools like ffmpeg and warn if missing."""
        if shutil.which("ffmpeg") is None:
            print("Warning: ffmpeg not found in PATH. Video writing may fail.")
            print("         On macOS you can install it with: brew install ffmpeg")

    def install_requirements(self) -> bool:
        self.print_header("Installing Python requirements")

        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            print(f"requirements.txt not found at {req_file}")
            return False

        print("Note: PyTorch is NOT installed by this script.")
        print("Install torch/torchvision/torchaudio manually from https://pytorch.org (e.g. 1.12.1 / 0.13.1 / 0.12.1).\n")

        ok = self.run_command("pip install -r requirements.txt", "Installing requirements")
        if ok:
            print("Requirements installed successfully.")
            # After installing, patch the local gfpgan package so that
            # its internal FaceRestoreHelper stores detection/parsing
            # weights under models/gfpgan/weights instead of a root
            # gfpgan/weights folder.
            self.patch_gfpgan_paths()
        return ok

    def patch_gfpgan_paths(self) -> None:
        """Best-effort patch of gfpgan.utils to honor models/ layout.

        This edits the installed gfpgan package in the current environment
        so that FaceRestoreHelper uses models/gfpgan/weights as its
        model_rootpath. If anything goes wrong, it just logs and continues.
        """
        try:
            import importlib

            gfpgan_spec = importlib.util.find_spec("gfpgan")
            if gfpgan_spec is None or not gfpgan_spec.origin:
                return

            utils_path = Path(gfpgan_spec.origin).with_name("utils.py")
            if not utils_path.exists():
                return

            text = utils_path.read_text(encoding="utf-8")
            old = "model_rootpath='gfpgan/weights'"
            new = "model_rootpath='models/gfpgan/weights'"
            if old not in text:
                return

            utils_path.write_text(text.replace(old, new), encoding="utf-8")
            print(f"Patched gfpgan utils to use models/gfpgan/weights: {utils_path}")
        except Exception as exc:
            print(f"Warning: could not patch gfpgan paths automatically: {exc}")

    def _download_file(self, url: str, dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            size_mb = dest.stat().st_size // (1024 * 1024)
            print(f"Already exists ({size_mb} MB): {dest}")
            return True

        print(f"Downloading:\n  URL : {url}\n  Dest: {dest}")
        try:
            with urllib.request.urlopen(url) as resp:
                total = resp.length or 0
                downloaded = 0
                chunk_size = 8192

                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded * 100.0 / total
                            bar_len = 30
                            filled = int(bar_len * percent / 100.0)
                            bar = "#" * filled + "-" * (bar_len - filled)
                            print(f"\r  [{bar}] {percent:5.1f}%", end="", flush=True)
                if total:
                    print()
        except Exception as exc:
            print(f"Failed to download {url}: {exc}")
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
            return False

        size_mb = dest.stat().st_size // (1024 * 1024)
        print(f"Downloaded {dest} ({size_mb} MB)")
        return True

    def download_models(self) -> bool:
        self.print_header("Downloading SadTalker checkpoints and enhancer weights")

        # Based on scripts/download_models.sh (new links)
        checkpoint_urls: Iterable[Tuple[str, Path]] = [
            (
                "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar",
                self.checkpoints_dir / "mapping_00109-model.pth.tar",
            ),
            (
                "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar",
                self.checkpoints_dir / "mapping_00229-model.pth.tar",
            ),
            (
                "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors",
                self.checkpoints_dir / "SadTalker_V0.0.2_256.safetensors",
            ),
            (
                "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors",
                self.checkpoints_dir / "SadTalker_V0.0.2_512.safetensors",
            ),
        ]

        enhancer_urls: Iterable[Tuple[str, Path]] = [
            (
                "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
                self.gfpgan_weights_dir / "alignment_WFLW_4HG.pth",
            ),
            (
                "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
                self.gfpgan_weights_dir / "detection_Resnet50_Final.pth",
            ),
            (
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                self.gfpgan_weights_dir / "GFPGANv1.4.pth",
            ),
            (
                "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
                self.gfpgan_weights_dir / "parsing_parsenet.pth",
            ),
        ]

        all_ok = True
        for url, dest in list(checkpoint_urls) + list(enhancer_urls):
            if not self._download_file(url, dest):
                all_ok = False

        if all_ok:
            print("All model files downloaded or already present.")
        else:
            print("Some model files failed to download. See messages above.")
        return all_ok

    def verify_setup(self) -> bool:
        self.print_header("Verifying SadTalker setup")

        required_paths = [
            ("requirements.txt", self.project_root / "requirements.txt"),
            ("models/checkpoints directory", self.checkpoints_dir),
            ("models/gfpgan/weights directory", self.gfpgan_weights_dir),
        ]

        ok = True
        for name, path in required_paths:
            if path.exists():
                print(f"✅ {name}: {path}")
            else:
                print(f"❌ {name} missing: {path}")
                ok = False

        # Specific key files
        key_files = [
            self.checkpoints_dir / "SadTalker_V0.0.2_256.safetensors",
            self.checkpoints_dir / "SadTalker_V0.0.2_512.safetensors",
            self.gfpgan_weights_dir / "GFPGANv1.4.pth",
        ]
        for f in key_files:
            label = f.name
            if f.exists():
                size_mb = f.stat().st_size // (1024 * 1024)
                print(f"✅ {label} ({size_mb} MB)")
            else:
                print(f"❌ {label} not found at {f}")
                ok = False

        return ok

    def run_complete_setup(self) -> bool:
        print("SadTalker COMPLETE SETUP")
        print("=" * 60)

        if not self.check_python_version():
            return False

        # Warn about missing system tools but do not hard-fail.
        self.check_system_tools()

        if not self.install_requirements():
            print("Failed to install Python requirements.")
            return False

        self.download_models()
        all_good = self.verify_setup()

        self.print_header("Setup summary")
        if all_good:
            print("SadTalker setup looks good. You can now run inference.")
        else:
            print("Setup is partial. See missing items above.")
        return all_good


def main() -> None:
    setup = SadTalkerSetup()

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--requirements-only":
            setup.install_requirements()
        elif arg == "--models-only":
            setup.download_models()
        elif arg == "--verify":
            setup.verify_setup()
        else:
            print("Usage: python setup.py [--requirements-only|--models-only|--verify]")
    else:
        setup.run_complete_setup()


if __name__ == "__main__":
    main()
