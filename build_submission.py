"""Build the root-only ZIP accepted by the automatic judge."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    root = Path(__file__).resolve().parent
    output = root / "dist" / "submission.zip"
    output.parent.mkdir(exist_ok=True)
    files = ("solution.py", "assistant.py", "corpus.py", "proxy.py", "requirements.txt")
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for filename in files:
            archive.write(root / filename, filename)
    print(output)


if __name__ == "__main__":
    main()
