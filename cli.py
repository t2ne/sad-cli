import os
import sys
import uuid
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def prompt_path(prompt: str) -> str:
    while True:
        value = input(f"{prompt}: ").strip().strip('"').strip("'")
        if value:
            p = Path(value).expanduser()
            if p.exists():
                return str(p)
            print(f"Path not found: {p}. Try again.")
        else:
            print("Please enter a non-empty path.")


def prompt_choice(prompt: str, choices: dict, default: str) -> str:
    print(prompt)
    for key, label in choices.items():
        print(f"  {key}) {label}")
    while True:
        value = input(f"Select option [{default}]: ").strip()
        if not value:
            return default
        if value in choices:
            return value
        print("Invalid choice, try again.")


def main() -> None:
    print("=== SadTalker CLI ===")
    print(f"Working directory: {PROJECT_ROOT}")

    image_path = prompt_path("Enter source image path (png/jpg/mp4)")
    audio_path = prompt_path("Enter driven audio path (wav/mp3/mp4)")

    preprocess_choice = prompt_choice(
        "Choose image preprocessing mode",
        {
            "1": "crop   (portrait / face crop)",
            "2": "full   (full image)",
            "3": "extfull (full image, extended crop)",
        },
        default="1",
    )

    if preprocess_choice == "1":
        preprocess = "crop"
    elif preprocess_choice == "2":
        preprocess = "full"
    else:
        preprocess = "extfull"

    size_choice = prompt_choice(
        "Choose output resolution",
        {
            "256": "256x256 (faster, less memory)",
            "512": "512x512 (slower, more memory)",
        },
        default="256",
    )
    size = size_choice

    enhancer_choice = prompt_choice(
        "Use face enhancer (GFPGAN)?",
        {
            "1": "No enhancer",
            "2": "Yes, GFPGAN",
        },
        default="1",
    )

    enhancer = "gfpgan" if enhancer_choice == "2" else None

    batch_size = "1"
    still = preprocess.startswith("full")

    result_root = PROJECT_ROOT / "results"
    result_root.mkdir(exist_ok=True)
    run_id = str(uuid.uuid4())
    result_dir = result_root / run_id

    cmd = [
        sys.executable,
        "inference.py",
        "--driven_audio",
        str(audio_path),
        "--source_image",
        str(image_path),
        "--result_dir",
        str(result_dir),
        "--size",
        size,
        "--batch_size",
        batch_size,
        "--preprocess",
        preprocess,
    ]

    if still:
        cmd.append("--still")

    if enhancer:
        cmd.extend(["--enhancer", enhancer])

    print("\nRunning SadTalker:")
    print(" ", " ".join(cmd))
    print()

    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))

    # inference.py writes result_dir/timestamp.mp4; search under this run's result_dir
    mp4_files = sorted(result_dir.rglob("*.mp4"), key=os.path.getmtime)
    if mp4_files:
        print(f"\nDone. Generated video: {mp4_files[-1]}")
    else:
        print(f"\nDone, but no mp4 found under {result_dir}. Check the results/ folder.")


if __name__ == "__main__":
    main()
