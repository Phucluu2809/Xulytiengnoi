"""Chuong trinh demo: doc nguong da huan luyen va xu ly tat ca WAV test."""
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt

from evaluation import analyze_file, arrange_four_figures, plot_result


ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "TinHieuKiemThu"
PARAMETERS_PATH = ROOT / "Results" / "best_parameters.json"


def load_threshold(path):
    """Doc nguong. Dau vao: duong dan JSON. Tra ve: threshold kieu float."""
    if not path.exists():
        raise FileNotFoundError("Chua co tham so. Hay chay train.py mot lan truoc khi demo.")
    with path.open("r", encoding="utf-8") as handle:
        parameters = json.load(handle)
    if "threshold" not in parameters:
        raise ValueError(f"Khong co threshold trong {path}")
    return float(parameters["threshold"])


def run_all_files(data_dir, threshold, save_dir=None):
    """Chay cac WAV. Dau vao: thu muc, nguong, noi luu. Tra ve: ds figure."""
    wav_files = sorted(data_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"Khong tim thay tep WAV trong {data_dir}")
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Moi WAV tao dung mot figure gom ket qua trung gian va ket qua cuoi.
    figures = []
    for wav_path in wav_files:
        result = analyze_file(wav_path, threshold)
        figure = plot_result(wav_path, threshold, result)
        figures.append(figure)
        snr_text = "N/A" if result["snr_db"] != result["snr_db"] else f"{result['snr_db']:.1f} dB"
        print(f"{wav_path.name}: {result['message']}, SNR={snr_text}")
        if save_dir is not None:
            figure.savefig(save_dir / f"{wav_path.stem}.png", dpi=150)
    return figures


def main(show=True, save_dir=None):
    """Chay demo. Dau vao: show, save_dir. Tra ve: khong co; hien 4 figure."""
    threshold = load_threshold(PARAMETERS_PATH)
    # Khi demo chi doc tin hieu test; tuyet doi khong dung lai WAV huan luyen.
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Khong ton tai thu muc kiem thu: {TEST_DIR}")
    print(f"Nguong da huan luyen: {threshold:.6f}")
    print(f"Du lieu kiem thu: {TEST_DIR}")

    # Tao tat ca figure trong mot lan Run va xep chung tren bon goc man hinh.
    figures = run_all_files(TEST_DIR, threshold, save_dir)
    arrange_four_figures(figures)
    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Khong mo figure")
    parser.add_argument("--save-dir", type=Path, help="Thu muc luu PNG")
    arguments = parser.parse_args()
    main(show=not arguments.no_show, save_dir=arguments.save_dir)
