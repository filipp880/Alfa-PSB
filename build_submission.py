"""Build a new numbered ZIP without overwriting past submissions."""

import argparse
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from corpus import INDEX_FILENAME, build_prebuilt_index

RUNTIME_FILES = ("solution.py", "assistant.py", "corpus.py", "fact_checks.py", "proxy.py", "telemetry.py",
                 "requirements.txt")
MAX_ZIP_BYTES = 128 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number", type=int, help="Номер архива. По умолчанию следующий свободный после последнего.")
    parser.add_argument("--kb", type=Path, help="База для предварительного индекса")
    parser.add_argument("--embedding-model", default="baai/bge-m3", help="ID модели в манифесте индекса")
    root = Path(__file__).resolve().parent
    args = parser.parse_args()
    folder = root / "submissions"
    folder.mkdir(exist_ok=True)
    latest = max((int(p.stem) for p in folder.glob("*.zip") if re.fullmatch(r"[1-9][0-9]*", p.stem)), default=0)
    number = args.number if args.number is not None else latest + 1
    if number < 1:
        parser.error("Номер должен быть положительным")
    output = folder / f"{number}.zip"
    if output.exists():
        parser.error(f"Архив {output.name} уже существует")
    index = build_prebuilt_index(args.kb or root / "data" / "knowledge-base.zip",
                                 embedding_model=args.embedding_model)
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for filename in (*RUNTIME_FILES, INDEX_FILENAME):
            entry = ZipInfo(filename)
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            contents = index if filename == INDEX_FILENAME else (root / filename).read_bytes()
            archive.writestr(entry, contents, compress_type=ZIP_DEFLATED, compresslevel=9)
    if buffer.tell() >= MAX_ZIP_BYTES:
        parser.error("Размер ZIP должен быть меньше 128 МБ")
    with ZipFile(buffer) as archive:
        if archive.testzip() is not None:
            parser.error("Нарушена целостность архива")
    try:
        with output.open("xb") as stream:
            stream.write(buffer.getbuffer())
    except FileExistsError:
        parser.error(f"Архив {output.name} уже существует")
    print(output)


if __name__ == "__main__":
    main()
