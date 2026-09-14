"""Huong luyen nguong nhi phan va luu tham so de dung khi demo."""
from pathlib import Path
import csv
import json
import numpy as np
from binary_search import find_threshold
from evaluation import analyze_file
from signal_processing import calculate_log_ma, labels_at_times, read_lab, read_wav


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "TinHieuHuanLuyen"
RESULTS_DIR = ROOT / "Results"
PARAMETERS_PATH = RESULTS_DIR / "best_parameters.json"
TRAINING_CSV_PATH = RESULTS_DIR / "training_results.csv"


def collect_training_values(train_dir):
    """Gom dac trung. Dau vao: thu muc train. Tra ve: log(MA) silence/speech."""
    silence_values, speech_values = [], []
    for wav_path in sorted(train_dir.glob("*.wav")):
        lab_path = wav_path.with_suffix(".lab")
        if not lab_path.exists():
            continue
        x, fs = read_wav(wav_path)
        times, feature = calculate_log_ma(x, fs)
        labels = labels_at_times(times, read_lab(lab_path))
        silence_values.append(feature[labels == 0])
        speech_values.append(feature[labels == 1])

    # Can it nhat mot cap WAV/LAB co du hai lop de huan luyen nguong.
    if not silence_values:
        raise FileNotFoundError(f"Khong tim thay cap WAV/LAB trong {train_dir}")
    return np.concatenate(silence_values), np.concatenate(speech_values)


def save_parameters(path, threshold, iterations):
    """Luu tham so. Dau vao: path, threshold, iterations. Tra ve: khong co."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = {"algorithm": "binary_search_log_ma", "threshold": threshold,
                  "frame_ms": 20.0, "hop_ms": 10.0,
                  "min_silence_ms": 300.0, "iterations": iterations}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(parameters, handle, ensure_ascii=False, indent=2)


def save_training_results(path, train_dir, threshold):
    """Luu CSV. Dau vao: path, train_dir, threshold. Tra ve: khong co."""
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "mae_ms", "rmse_ms", "status"])
        for wav_path in sorted(train_dir.glob("*.wav")):
            result = analyze_file(wav_path, threshold)
            writer.writerow([wav_path.name, result["mae"], result["rmse"],
                             result["message"]])


def main():
    """Chay training. Dau vao: khong co. Tra ve: khong co; tao JSON va CSV."""
    silence, speech = collect_training_values(TRAIN_DIR)
    threshold, iterations = find_threshold(silence, speech)
    save_parameters(PARAMETERS_PATH, threshold, iterations)
    save_training_results(TRAINING_CSV_PATH, TRAIN_DIR, threshold)
    print(f"Da luu nguong T={threshold:.6f} vao {PARAMETERS_PATH}")
    print(f"Da luu ket qua huan luyen vao {TRAINING_CSV_PATH}")


if __name__ == "__main__":
    main()
