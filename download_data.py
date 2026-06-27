import subprocess
import sys
from pathlib import Path

DATASET = "gokulrajkmv/unemployment-in-india"
DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-p",
            str(DATA_DIR),
            "--unzip",
        ],
        check=True,
    )
    print(f"Downloaded to {DATA_DIR}")


if __name__ == "__main__":
    main()
