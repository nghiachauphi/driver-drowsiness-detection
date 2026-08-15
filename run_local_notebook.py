"""Run the training notebook portably on Windows, macOS, or Linux."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import kagglehub
import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = ROOT / "DDD_Drowsiness_Executed.ipynb"
DATASET_HANDLE = "ismailnasri20/driver-drowsiness-dataset-ddd"


def configure_console() -> None:
    """Keep Vietnamese messages printable on older Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def configure_kernel_environment() -> None:
    """Make the Jupyter kernel resolve to the interpreter running this script."""
    interpreter_dir = str(Path(sys.executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = interpreter_dir + os.pathsep + current_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo và chạy notebook huấn luyện cục bộ."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Thư mục dataset đã giải nén; bỏ qua để KaggleHub tự tải.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Dùng toàn bộ dữ liệu. Mặc định là pilot 2%% để chạy thử nhanh.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Huấn luyện lại dù outputs/results đã có kết quả cũ.",
    )
    parser.add_argument(
        "--output-notebook",
        type=Path,
        default=ROOT / "DDD_Drowsiness_Local.ipynb",
        help="Nơi lưu notebook đã chạy.",
    )
    return parser.parse_args()


def resolve_dataset(dataset: Path | None) -> Path:
    if dataset is not None:
        path = dataset.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {path}")
        return path

    print(f"Đang tải/tìm dataset Kaggle: {DATASET_HANDLE}")
    return Path(kagglehub.dataset_download(DATASET_HANDLE)).resolve()


def prepare_notebook(
    source: Path, dataset: Path, *, pilot: bool, force: bool
) -> nbformat.NotebookNode:
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy notebook nguồn: {source}")

    notebook = nbformat.read(source, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        source_code = cell.source

        if "PROJECT_DIR =" in source_code and "OUTPUT_DIR" in source_code:
            cell.source = f"""import os
PROJECT_DIR = {str(ROOT)!r}
OUTPUT_DIR  = os.path.join(PROJECT_DIR, 'outputs')
FIG_DIR     = os.path.join(OUTPUT_DIR, 'figures')
MODEL_DIR   = os.path.join(OUTPUT_DIR, 'models')
TABLE_DIR   = os.path.join(OUTPUT_DIR, 'tables')
RESULT_DIR  = os.path.join(OUTPUT_DIR, 'results')
for d in (FIG_DIR, MODEL_DIR, TABLE_DIR, RESULT_DIR):
    os.makedirs(d, exist_ok=True)
print('Thư mục kết quả:', OUTPUT_DIR)"""

        elif re.search(r"^ds_path\s*=", source_code, flags=re.MULTILINE):
            cell.source = (
                f"ds_path = {str(dataset)!r}\n"
                "print('Dataset:', ds_path)"
            )

        elif re.search(r"^PILOT\s*=", source_code, flags=re.MULTILINE):
            source_code = re.sub(
                r"^PILOT\s*=.*$",
                f"PILOT = {pilot}   # True = pilot 2%; False = toàn bộ dữ liệu",
                source_code,
                count=1,
                flags=re.MULTILINE,
            )
            cell.source = source_code.replace(
                "CFG = {", f"FORCE_RETRAIN = {force}\n\nCFG = {{", 1
            )

        elif "if os.path.exists(rf):" in source_code:
            cell.source = source_code.replace(
                "if os.path.exists(rf):",
                "if os.path.exists(rf) and not FORCE_RETRAIN:",
            )

        elif "Google Colab" in source_code and "env_rows" in source_code:
            cell.source = source_code.replace("'Google Colab'", "'Máy cục bộ'")

        elif "Vào Runtime → Change runtime type" in source_code:
            cell.source = source_code.replace(
                "Vào Runtime → Change runtime type → T4 GPU.",
                "Đang huấn luyện bằng CPU; chế độ đầy đủ có thể mất nhiều giờ.",
            )

        cell.outputs = []
        cell.execution_count = None

    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return notebook


def main() -> None:
    configure_console()
    configure_kernel_environment()
    args = parse_args()
    dataset = resolve_dataset(args.dataset)
    target = args.output_notebook.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    notebook = prepare_notebook(
        SOURCE_NOTEBOOK,
        dataset,
        pilot=not args.full,
        force=args.force,
    )
    nbformat.write(notebook, target)

    print(f"Bắt đầu huấn luyện: {'đầy đủ' if args.full else 'pilot 2%'}")
    print(f"Notebook kết quả: {target}")
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, target)

    model = ROOT / "outputs" / "models" / "E6_MobileNetV2_subject.keras"
    if not model.is_file():
        raise FileNotFoundError(f"Huấn luyện xong nhưng chưa thấy mô hình E6: {model}")
    print(f"Hoàn tất. Mô hình dùng cho Streamlit: {model}")


if __name__ == "__main__":
    main()
