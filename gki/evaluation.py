"""Cac ham tao nhan du doan, danh gia bien va ve ket qua."""

import matplotlib.pyplot as plt
import numpy as np

from signal_processing import (calculate_log_ma, labels_at_times, read_lab,
                               read_wav, remove_short_silences)


def boundaries_from_labels(times, labels):
    """Tim bien du doan. Dau vao: times, labels. Tra ve: cac thoi diem doi nhan."""
    if labels.size < 2:
        return np.array([])
    return times[1:][labels[1:] != labels[:-1]]


def ground_truth_boundaries(segments):
    """Tim bien chuan. Dau vao: cac segment LAB. Tra ve: bien sil va v/uv."""
    states = [label != "sil" for _, _, label in segments]
    return np.array([segments[k][0] for k in range(1, len(segments))
                     if states[k] != states[k - 1]], dtype=float)


def boundary_errors(reference, predicted):
    """Danh gia bien. Dau vao: bien chuan/du doan. Tra ve: MAE va RMSE (ms)."""
    if reference.size == 0:
        return float("nan"), float("nan"), "Khong co ground truth"
    if predicted.size == 0:
        return float("nan"), float("nan"), "Khong tim duoc bien du doan"

    # Ghep moi bien chuan voi bien du doan gan nhat de tinh sai so thoi gian.
    errors = np.array([np.min(np.abs(predicted - value)) for value in reference]) * 1000
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return mae, rmse, f"MAE={mae:.1f} ms, RMSE={rmse:.1f} ms"


def estimate_snr_db(x, fs, segments):
    """Uoc luong SNR. Dau vao: x, fs, LAB. Tra ve: SNR speech/noise (dB)."""
    if not segments or x.size == 0:
        return float("nan")
    speech_mask = np.zeros(x.size, dtype=bool)
    silence_mask = np.zeros(x.size, dtype=bool)

    # Doi cac moc thoi gian LAB sang mien mau de tach cong suat hai lop.
    for left, right, label in segments:
        start = max(0, int(round(left * fs)))
        end = min(x.size, int(round(right * fs)))
        if label == "sil":
            silence_mask[start:end] = True
        elif label in ("v", "uv"):
            speech_mask[start:end] = True
    if not np.any(speech_mask) or not np.any(silence_mask):
        return float("nan")

    # SNR = 10log10(Pspeech/Pnoise); chi dung de danh gia, khong de du doan.
    speech_power = np.mean(x[speech_mask] ** 2)
    noise_power = np.mean(x[silence_mask] ** 2)
    if noise_power <= 0:
        return float("inf")
    return float(10.0 * np.log10(speech_power / noise_power))


def analyze_file(wav_path, threshold):
    """Phan tich WAV. Dau vao: wav_path, threshold. Tra ve: dictionary ket qua."""
    x, fs = read_wav(wav_path)
    times, feature = calculate_log_ma(x, fs)
    raw_labels = (feature >= threshold).astype(np.uint8)
    labels = remove_short_silences(raw_labels)
    predicted = boundaries_from_labels(times, labels)

    # LAB chi dung de danh gia sau khi thuat toan da dua ra ket qua.
    lab_path = wav_path.with_suffix(".lab")
    segments = read_lab(lab_path) if lab_path.exists() else []
    reference_labels = labels_at_times(times, segments)
    reference = ground_truth_boundaries(segments)
    mae, rmse, message = boundary_errors(reference, predicted)
    snr_db = estimate_snr_db(x, fs, segments)
    return {"x": x, "fs": fs, "times": times, "feature": feature,
            "labels": labels, "predicted": predicted, "reference": reference,
            "reference_labels": reference_labels,
            "mae": mae, "rmse": rmse, "snr_db": snr_db, "message": message}


def plot_result(wav_path, threshold, result, container=None, show_legend=True):
    """Ve day du ket qua mot WAV, doc lap hoac trong subfigure tong hop."""
    signal_time = np.arange(result["x"].size) / result["fs"]
    if container is None:
        fig, axes = plt.subplots(3, 1, num=wav_path.stem, figsize=(9, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2.0, 1.25, 0.8]})
    else:
        fig = container
        axes = container.subplots(3, 1, sharex=True,
                                  gridspec_kw={"height_ratios": [2.0, 1.25, 0.8]})
    axes[0].plot(signal_time, result["x"], color="0.25", linewidth=0.7,
                 label="Waveform")
    axes[0].set(title="1. Tin hieu dau vao va cac bien", xlabel="Thoi gian (s)",
                ylabel="Bien do chuan hoa")
    axes[0].grid(alpha=0.2)

    # Xep chong log(MA) len waveform bang truc Y thu hai theo dung yeu cau de bai.
    feature_axis = axes[0].twinx()
    feature_axis.plot(result["times"], result["feature"], color="darkorange",
                      linewidth=0.8, alpha=0.55, label="log(MA)")
    feature_axis.set_ylabel("log(MA)", color="darkorange")
    feature_axis.tick_params(axis="y", colors="darkorange")

    # Subplot thu hai chi trinh bay ket qua trung gian: log(MA) va nguong T.
    feature = result["feature"]
    times = result["times"]
    axes[1].plot(times, feature, color="black", linewidth=1, label="log(MA)")
    axes[1].axhline(threshold, color="darkorange", linestyle="--",
                    label=f"T = {threshold:.4f}")
    axes[1].set(title="2. Ket qua trung gian: dac trung log(MA) va nguong",
                xlabel="Thoi gian (s)", ylabel="log(MA)")
    axes[1].grid(alpha=0.2)
    if show_legend:
        axes[1].legend(loc="upper right", fontsize=8)

    # Subplot thu ba tach rieng ket qua cuoi cung de de so sanh voi nhan LAB.
    axes[2].step(times, result["labels"], where="mid", color="blue",
                 linewidth=1.4, label="Du doan")
    if result["reference"].size:
        axes[2].step(times, result["reference_labels"], where="mid", color="red",
                     linestyle="--", linewidth=1.2, label="Ground truth")
    axes[2].set(title="3. Ket qua cuoi: phan doan speech/silence",
                xlabel="Thoi gian (s)", ylabel="Nhan", yticks=[0, 1])
    axes[2].set_yticklabels(["Silence", "Speech"])
    axes[2].set_ylim(-0.15, 1.15)
    axes[2].grid(alpha=0.2)
    if show_legend:
        axes[2].legend(loc="upper right", fontsize=8)

    # Ve bien du doan va ground-truth tren waveform de danh gia truc quan.
    for axis in (axes[0],):
        for index, boundary in enumerate(result["predicted"]):
            axis.axvline(boundary, color="blue", linewidth=1,
                         label="Bien du doan" if index == 0 else None)
        for index, boundary in enumerate(result["reference"]):
            axis.axvline(boundary, color="red", linestyle="--", linewidth=1,
                         label="Bien chuan" if index == 0 else None)
        waveform_lines, waveform_names = axis.get_legend_handles_labels()
        feature_lines, feature_names = feature_axis.get_legend_handles_labels()
        if show_legend:
            axis.legend(waveform_lines + feature_lines, waveform_names + feature_names,
                        loc="upper right", fontsize=8)
    snr_text = "SNR=N/A" if np.isnan(result["snr_db"]) else f"SNR={result['snr_db']:.1f} dB"
    fig.suptitle(f"{wav_path.name} | {result['message']} | {snr_text}")
    if container is None:
        fig.tight_layout()
    return fig


def arrange_four_figures(figures):
    """Xep figure. Dau vao: toi da 4 figure. Tra ve: khong co."""
    if not figures:
        return
    try:
        manager = figures[0].canvas.manager
        screen_w = manager.window.winfo_screenwidth()
        screen_h = manager.window.winfo_screenheight()
        width, height = screen_w // 2, screen_h // 2
        positions = [(0, 0), (width, 0), (0, height), (width, height)]
        for figure, (left, top) in zip(figures[:4], positions):
            figure.canvas.manager.window.wm_geometry(f"{width}x{height}+{left}+{top}")
    except (AttributeError, TypeError):
        pass  # Backend khong ho tro dieu chinh vi tri cua so.
