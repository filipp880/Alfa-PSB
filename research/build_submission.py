"""Build a new numbered ZIP without overwriting past submissions."""

import argparse
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

from corpus import INDEX_FILENAME, build_prebuilt_index

RUNTIME_FILES = ("solution.py", "assistant.py", "corpus.py", "fact_checks.py", "proxy.py", "telemetry.py",
                 "requirements.txt")
MAX_ZIP_BYTES = 128 * 1024 * 1024


def archive_files(root: Path, args: argparse.Namespace, number: int) -> dict[str, bytes]:
    base, overlay = args.base, args.overlay
    if number == 9 and base is None:
        base = 3
    if number == 9 and overlay is None and base == 3:
        overlay = root / "submission9"
    if base is None:
        if overlay is not None:
            raise ValueError("--overlay требует --base")
        index = build_prebuilt_index(args.kb or root / "data" / "knowledge-base.zip",
                                     embedding_model=args.embedding_model)
        return {**{name: (root / name).read_bytes() for name in RUNTIME_FILES}, INDEX_FILENAME: index}

    with ZipFile(root / "submissions" / f"{base}.zip") as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or not {"solution.py", "requirements.txt"} <= set(names)
                or any(Path(name).name != name or name not in (*RUNTIME_FILES, INDEX_FILENAME) for name in names)):
            raise ValueError("Неподдерживаемые файлы в исходном сабмите")
        files = {name: archive.read(name) for name in names}
    if overlay is not None:
        if not overlay.is_dir():
            raise ValueError(f"Каталог исходников отсутствует: {overlay}")
        # Keep ZIP bytes reproducible across Git checkouts with different line endings.
        patches = {path.name: path.read_bytes().replace(b"\r\n", b"\n")
                   for path in overlay.glob("*.py") if path.is_file()}
        if not patches or not patches.keys() <= files.keys():
            raise ValueError("В каталоге изменений могут быть только модули исходного сабмита")
        if number == 9 and not {"assistant.py", "solution.py", "proxy.py"} <= patches.keys():
            raise ValueError("Для № 9 нужны дополнения assistant.py, solution.py и proxy.py")
        files.update(patches)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number", type=int, help="Номер архива. По умолчанию следующий свободный после последнего.")
    parser.add_argument("--kb", type=Path, help="База для предварительного индекса")
    parser.add_argument("--embedding-model", default="baai/bge-m3", help="ID модели в манифесте индекса")
    parser.add_argument("--base", type=int, help="Использовать файлы указанного сабмита")
    parser.add_argument("--overlay", type=Path, help="Заменить корневые модули исходного сабмита")
    parser.add_argument("--output-dir", type=Path, help="Каталог для контрольной сборки")
    root = Path(__file__).resolve().parent
    args = parser.parse_args()
    folder = args.output_dir or root / "submissions"
    folder.mkdir(exist_ok=True)
    latest = max((int(p.stem) for p in (root / "submissions").glob("*.zip")
                  if re.fullmatch(r"[1-9][0-9]*", p.stem)), default=0)
    number = args.number if args.number is not None else latest + 1
    if number < 1:
        parser.error("Номер должен быть положительным")
    output = folder / f"{number}.zip"
    if output.exists():
        parser.error(f"Архив {output.name} уже существует")
    try:
        files = archive_files(root, args, number)
    except (OSError, ValueError, BadZipFile) as exc:
        parser.error(str(exc))
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for filename, contents in files.items():
            entry = ZipInfo(filename)
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
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
