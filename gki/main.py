"""Chuong trinh demo: doc nguong da huan luyen va xu ly tat ca WAV test."""
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from evaluation import analyze_file


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
    """Chay 4 WAV va ve 4 bieu do day du trong mot figure 2x2."""
    wav_files = sorted(data_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"Khong tim thay tep WAV trong {data_dir}")
    if len(wav_files) != 4:
        raise ValueError(f"Can dung 4 tep WAV; tim thay {len(wav_files)} tep")
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, num="Ket qua 4 tin hieu kiem thu",
                                figsize=(16, 10), layout="constrained")
    for axis, wav_path in zip(axes.flat, wav_files):
        result = analyze_file(wav_path, threshold)
        signal_time = np.arange(result["x"].size) / result["fs"]

        # Input va dac trung log(MA) xep chong tren cung truc thoi gian.
        axis.plot(signal_time, result["x"], color="0.35", linewidth=0.65,
                  label="Tin hieu WAV")
        feature_axis = axis.twinx()
        feature_axis.plot(result["times"], result["feature"],
                          color="darkorange", linewidth=0.85, alpha=0.75,
                          label="log(MA)")
        feature_axis.axhline(threshold, color="purple", linestyle=":",
                             linewidth=1.1, label=f"Nguong T={threshold:.4f}")

        # Output thuat toan va ground truth duoc ve truc tiep tren input.
        for index, boundary in enumerate(result["predicted"]):
            axis.axvline(boundary, color="blue", linewidth=1.1,
                         label="Bien du doan" if index == 0 else None)
        for index, boundary in enumerate(result["reference"]):
            axis.axvline(boundary, color="red", linestyle="--", linewidth=1.1,
                         label="Bien ground truth" if index == 0 else None)

        snr_text = "N/A" if result["snr_db"] != result["snr_db"] else f"{result['snr_db']:.1f} dB"
        mae_text = "N/A" if np.isnan(result["mae"]) else f"{result['mae']:.1f} ms"
        rmse_text = "N/A" if np.isnan(result["rmse"]) else f"{result['rmse']:.1f} ms"
        axis.set_title(f"{wav_path.name}\nMAE={mae_text} | RMSE={rmse_text} | SNR={snr_text}",
                       fontsize=10)
        axis.set_xlabel("Thoi gian (s)")
        axis.set_ylabel("Bien do chuan hoa")
        feature_axis.set_ylabel("log(MA)", color="darkorange")
        feature_axis.tick_params(axis="y", colors="darkorange")
        axis.set_xlim(0, signal_time[-1] if signal_time.size else 1)
        axis.grid(alpha=0.2)
        print(f"{wav_path.name}: {result['message']}, SNR={snr_text}")

    # Mot bang chu giai duy nhat cho toan bo bon bieu do.
    handles_by_label = {}
    for current_axis in figure.axes:
        handles, labels = current_axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_"):
                handles_by_label.setdefault(label, handle)
    figure.legend(handles_by_label.values(), handles_by_label.keys(),
                  loc="outside upper center", ncol=5, fontsize=9,
                  frameon=True)
    figure.suptitle("Phan doan speech/silence tren 4 tin hieu kiem thu", fontsize=14)
    if save_dir is not None:
        figure.savefig(save_dir / "four_results.png", dpi=160)
    return figure


def main(show=True, save_dir=None):
    """Chay demo mot lan va hien mot figure gom 4 bieu do WAV."""
    if not show:
        plt.switch_backend("Agg")
    threshold = load_threshold(PARAMETERS_PATH)
    # Khi demo chi doc tin hieu test; tuyet doi khong dung lai WAV huan luyen.
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Khong ton tai thu muc kiem thu: {TEST_DIR}")
    print(f"Nguong da huan luyen: {threshold:.6f}")
    print(f"Du lieu kiem thu: {TEST_DIR}")

    # Mot lan chay tao mot figure gom bon tin hieu kiem thu.
    run_all_files(TEST_DIR, threshold, save_dir)
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
