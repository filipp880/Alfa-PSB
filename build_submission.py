"""Build a new numbered ZIP without overwriting past submissions."""

import argparse
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

RUNTIME_FILES = ("solution.py", "assistant.py", "corpus.py", "proxy.py", "telemetry.py", "requirements.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number", type=int, help="Номер архива. По умолчанию следующий свободный после последнего.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    folder = root / "submissions"
    folder.mkdir(exist_ok=True)
    latest = max((int(p.stem) for p in folder.glob("*.zip") if re.fullmatch(r"[1-9][0-9]*", p.stem)), default=0)
    number = args.number if args.number is not None else latest + 1
    if number < 1:
        parser.error("Номер должен быть положительным")
    output = folder / f"{number}.zip"
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for filename in RUNTIME_FILES:
            entry = ZipInfo(filename)
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, (root / filename).read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)
    try:
        with output.open("xb") as stream:
            stream.write(buffer.getbuffer())
    except FileExistsError:
        parser.error(f"Архив {output.name} уже существует")
    print(output)


if __name__ == "__main__":
    main()
