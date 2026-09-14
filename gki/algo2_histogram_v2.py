# -*- coding: utf-8 -*-
"""
THUAT TOAN 2 - LOAI KHOANG LANG VA PHAN DOAN TIENG NOI BANG HISTOGRAM.

Pipeline:
  1. Chia tin hieu thanh cac frame 50 ms khong chong lap.
  2. Tu tinh STE va spectral centroid bang DFT.
  3. Tu tao histogram, lam muot, tim hai cuc dai M1/M2 va tinh nguong.
  4. Khung speech khi E(i) > T1 VA C(i) > T2.
  5. Mo rong moi doan speech 5 frame ve hai phia, sau do hop nhat.
  6. Ve 4 figure test, bien thuat toan/ground-truth va MAE/RMSE.

Khong dung scipy hay cac ham toolbox xu ly tin hieu. Cac phep tinh co ban
duoc viet bang Python/NumPy; matplotlib chi duoc dung de hien thi ket qua.
"""

import itertools
import os
import wave

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# 0. CAU HINH CHUONG TRINH
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "TinHieuHuanLuyen")
TEST_DIR = os.path.join(BASE_DIR, "TinHieuKiemThu")
OUT_DIR = os.path.join(BASE_DIR, "Results")

FRAME_MS = 50                 # Bat buoc theo de bai: frame 50 ms.
EXTEND_FRAMES = 5             # 5 frame = 250 ms o moi dau segment.
EPS = 1e-12

# Training chi chon tham so cua buoc histogram; FRAME_MS khong duoc thay doi.
N_HIST_BINS_CANDIDATES = [15, 20, 25, 30]
SMOOTH_WIN_CANDIDATES = [3, 5, 7]
W_ENERGY_CANDIDATES = [3, 5, 7]
W_CENTROID_CANDIDATES = [3, 5, 7]


# ============================================================================
# 1. DOC WAV, NHAN .LAB VA TIM CAP TIN HIEU
# ============================================================================

def load_wav(path):
    """Doc WAV PCM; tra ve (tin_hieu_mono_float, tan_so_lay_mau_Hz)."""
    with wave.open(path, "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw_bytes = wav_file.readframes(n_frames)

    # Giai ma mau PCM ma khong dung scipy.io.wavfile.
    if sample_width == 1:
        samples = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float64)
        samples = samples - 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float64)
    elif sample_width == 3:
        byte_data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        samples = (byte_data[:, 0].astype(np.int32)
                   | (byte_data[:, 1].astype(np.int32) << 8)
                   | (byte_data[:, 2].astype(np.int32) << 16))
        samples = np.where(samples & 0x800000, samples - 0x1000000, samples)
        samples = samples.astype(np.float64)
    elif sample_width == 4:
        samples = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float64)
    else:
        raise ValueError(f"Khong ho tro WAV {sample_width * 8}-bit: {path}")

    # Chuyen da kenh thanh mono va chuan hoa ve [-1, 1].
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
        samples = np.sum(samples, axis=1) / n_channels
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak > EPS:
        samples = samples / peak
    return samples, sample_rate


def parse_lab(path):
    """Doc .lab; tra ve cac doan va bien silence/speech theo don vi giay."""
    segments = []
    with open(path, "r", encoding="utf-8") as lab_file:
        for line in lab_file:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            try:
                start, end = float(parts[0]), float(parts[1])
            except ValueError:       # Bo qua F0mean, F0std o cuoi file train.
                continue
            label = "sil" if parts[2].lower() == "sil" else "speech"
            if segments and segments[-1][2] == label:
                segments[-1][1] = end   # Gop v va uv thanh mot mien speech.
            else:
                segments.append([start, end, label])

    boundaries = [segments[i][1] for i in range(len(segments) - 1)]
    return segments, boundaries


def find_signal_pairs(folder):
    """Tra ve danh sach ten goc cua cac cap .wav + .lab trong folder."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Khong tim thay thu muc:\n{folder}")
    names = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".wav"):
            name = os.path.splitext(filename)[0]
            if os.path.isfile(os.path.join(folder, name + ".lab")):
                names.append(name)
    return sorted(names)


# ============================================================================
# 2. CHIA FRAME VA TU TINH HAI CHUOI DAC TRUNG
# ============================================================================

def frame_signal(signal, sample_rate, frame_ms=FRAME_MS):
    """Chia frame khong chong lap; tra ve frames, tam frame va do dai frame."""
    frame_len = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    n_frames = int(np.ceil(len(signal) / frame_len))
    padded = np.zeros(n_frames * frame_len, dtype=np.float64)
    padded[:len(signal)] = signal
    frames = padded.reshape(n_frames, frame_len)
    times = (np.arange(n_frames) + 0.5) * frame_len / sample_rate
    return frames, times, frame_len / sample_rate


def manual_rfft_magnitude(frames):
    """Tu tinh bien do DFT mot phia cho tung frame; khong dung np.fft."""
    frame_len = frames.shape[1]
    k = np.arange(frame_len // 2 + 1, dtype=np.float64)[:, None]
    n = np.arange(frame_len, dtype=np.float64)[None, :]
    angle = 2.0 * np.pi * k * n / frame_len

    # X(k) = sum x(n)[cos(2*pi*k*n/N) - j*sin(2*pi*k*n/N)].
    real_part = frames @ np.cos(angle).T
    imag_part = -(frames @ np.sin(angle).T)
    magnitude = np.sqrt(real_part * real_part + imag_part * imag_part)
    return magnitude


def compute_features(signal, sample_rate, frame_ms=FRAME_MS):
    """Tra ve STE E(i), spectral centroid C(i), moc thoi gian va frame duration."""
    frames, times, frame_dur = frame_signal(signal, sample_rate, frame_ms)
    frame_len = frames.shape[1]

    # E(i) = (1/N) * sum |x_i(n)|^2.
    energy = np.sum(frames * frames, axis=1) / frame_len
    magnitude = manual_rfft_magnitude(frames)
    bin_indices = np.arange(1, magnitude.shape[1] + 1, dtype=np.float64)
    denominator = np.sum(magnitude, axis=1)
    numerator = np.sum(magnitude * bin_indices[None, :], axis=1)
    centroid = np.where(denominator > EPS, numerator / denominator, 0.0)
    return energy, centroid, times, frame_dur


# ============================================================================
# 3. TU CAI DAT HISTOGRAM, LAM MUOT, CUC DAI VA NGUONG DONG
# ============================================================================

def manual_histogram(values, n_bins):
    """Tu dem histogram; tra ve tan su va tam cua n_bins khoang gia tri."""
    low, high = float(np.min(values)), float(np.max(values))
    if high - low <= EPS:
        return np.array([len(values)], dtype=float), np.array([low])
    width = (high - low) / n_bins
    counts = np.zeros(n_bins, dtype=np.float64)
    for value in values:
        index = min(int((float(value) - low) / width), n_bins - 1)
        counts[index] += 1.0
    centers = low + (np.arange(n_bins) + 0.5) * width
    return counts, centers


def moving_average_smooth(values, window_size):
    """Tu lam muot moving-average; tra ve chuoi cung kich thuoc dau vao."""
    if window_size <= 1:
        return values.astype(float).copy()
    radius = window_size // 2
    output = np.zeros(len(values), dtype=np.float64)
    for i in range(len(values)):
        left, right = max(0, i - radius), min(len(values), i + radius + 1)
        output[i] = np.sum(values[left:right]) / (right - left)
    return output


def find_local_maxima(values):
    """Tu tim chi so cuc dai cuc bo, ke ca hai dau cua histogram."""
    peaks = []
    for i in range(len(values)):
        left = values[i - 1] if i > 0 else -np.inf
        right = values[i + 1] if i + 1 < len(values) else -np.inf
        if values[i] >= left and values[i] >= right and values[i] > 0:
            peaks.append(i)
    return peaks


def histogram_threshold(values, n_bins, smooth_win, weight):
    """Tinh T=(W*M1+M2)/(W+1); tra them histogram va hai dinh de ve."""
    hist, centers = manual_histogram(values, n_bins)
    smoothed = moving_average_smooth(hist, smooth_win)
    peak_indices = find_local_maxima(smoothed)

    # Lay hai cuc dai cao nhat; fallback dung hai nua histogram/median.
    if len(peak_indices) >= 2:
        ranked = sorted(peak_indices, key=lambda idx: smoothed[idx], reverse=True)
        chosen = sorted(ranked[:2])
        m1, m2 = float(centers[chosen[0]]), float(centers[chosen[1]])
    else:
        midpoint = float(np.median(values))
        lower = values[values <= midpoint]
        upper = values[values > midpoint]
        m1 = float(np.mean(lower)) if len(lower) else midpoint
        m2 = float(np.mean(upper)) if len(upper) else midpoint
    threshold = (weight * m1 + m2) / (weight + 1.0)
    return threshold, centers, smoothed, (m1, m2)


# ============================================================================
# 4. QUYET DINH SPEECH VA HAU XU LY ±5 FRAME
# ============================================================================

def binary_frames_to_speech_segments(decisions, frame_dur, total_dur):
    """Doi cac day frame True lien tiep thanh cac doan speech [start, end]."""
    segments = []
    start_index = None
    for i, is_speech in enumerate(decisions):
        if is_speech and start_index is None:
            start_index = i
        is_last = i == len(decisions) - 1
        if start_index is not None and ((not is_speech) or is_last):
            end_index = i + 1 if is_speech and is_last else i
            segments.append([start_index * frame_dur,
                             min(end_index * frame_dur, total_dur)])
            start_index = None
    return segments


def extend_and_merge_segments(segments, total_dur, frame_dur,
                              extend_frames=EXTEND_FRAMES):
    """Mo rong moi speech segment ±extend_frames roi hop cac doan giao/sat nhau."""
    margin = extend_frames * frame_dur
    extended = [[max(0.0, start - margin), min(total_dur, end + margin)]
                for start, end in segments]
    if not extended:
        return []

    merged = [extended[0]]
    for start, end in extended[1:]:
        if start <= merged[-1][1] + EPS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def boundaries_from_speech_segments(segments, total_dur):
    """Lay bien speech/silence ben trong tin hieu, bo bien tai 0 va cuoi file."""
    boundaries = []
    for start, end in segments:
        if start > EPS:
            boundaries.append(start)
        if end < total_dur - EPS:
            boundaries.append(end)
    return sorted(boundaries)


# ============================================================================
# 5. DANH GIA BIEN VA KHAO SAT SNR
# ============================================================================

def evaluate_boundaries_ms(detected, groundtruth):
    """Ghep moi bien GT voi bien detected gan nhat; tra ve RMSE, MAE (ms)."""
    if not detected or not groundtruth:
        return None, None, np.array([])
    errors = []
    for gt_boundary in groundtruth:
        nearest = min(detected, key=lambda value: abs(value - gt_boundary))
        errors.append(abs(nearest - gt_boundary) * 1000.0)
    errors = np.asarray(errors, dtype=np.float64)
    rmse = float(np.sqrt(np.sum(errors * errors) / len(errors)))
    mae = float(np.sum(errors) / len(errors))
    return rmse, mae, errors


def estimate_snr_db(energy, decisions):
    """Uoc luong SNR tu STE cua cac frame speech/silence do thuat toan tim."""
    speech_energy = energy[decisions]
    noise_energy = energy[~decisions]
    if len(speech_energy) == 0 or len(noise_energy) == 0:
        return None
    speech_power = float(np.sum(speech_energy) / len(speech_energy))
    noise_power = float(np.sum(noise_energy) / len(noise_energy))
    ratio = max((speech_power - noise_power) / (noise_power + EPS), EPS)
    return float(10.0 * np.log10(ratio))


# ============================================================================
# 6. CHAY PIPELINE CHO MOT FILE
# ============================================================================

def extract_file_base(name, data_dir):
    """Doc file va tinh dac trung co dinh 50 ms de tai su dung khi training."""
    signal, sample_rate = load_wav(os.path.join(data_dir, name + ".wav"))
    gt_segments, gt_boundaries = parse_lab(os.path.join(data_dir, name + ".lab"))
    energy, centroid, times, frame_dur = compute_features(
        signal, sample_rate, FRAME_MS
    )
    return {
        "name": name, "signal": signal, "sample_rate": sample_rate,
        "total_dur": len(signal) / sample_rate,
        "time_signal": np.arange(len(signal)) / sample_rate,
        "gt_segments": gt_segments, "gt_boundaries": gt_boundaries,
        "energy": energy, "centroid": centroid,
        "times": times, "frame_dur": frame_dur,
    }


def apply_pipeline(base, params):
    """Ap dung nguong, phep AND, mo rong/hop nhat va danh gia tren dac trung."""
    t_energy, bins_e, hist_e, peaks_e = histogram_threshold(
        base["energy"], params["N_HIST_BINS"], params["SMOOTH_WIN"],
        params["W_ENERGY"]
    )
    t_centroid, bins_c, hist_c, peaks_c = histogram_threshold(
        base["centroid"], params["N_HIST_BINS"], params["SMOOTH_WIN"],
        params["W_CENTROID"]
    )

    # Dung dung tieu chi: ca hai chuoi dac trung cung vuot nguong.
    decisions = (base["energy"] > t_energy) & (base["centroid"] > t_centroid)
    raw_segments = binary_frames_to_speech_segments(
        decisions, base["frame_dur"], base["total_dur"]
    )
    speech_segments = extend_and_merge_segments(
        raw_segments, base["total_dur"], base["frame_dur"], EXTEND_FRAMES
    )
    boundaries = boundaries_from_speech_segments(speech_segments, base["total_dur"])
    rmse, mae, errors = evaluate_boundaries_ms(boundaries, base["gt_boundaries"])

    result = dict(base)
    result.update({
        "T_energy": t_energy, "bins_e": bins_e, "hist_e": hist_e,
        "peaks_e": peaks_e, "T_centroid": t_centroid,
        "bins_c": bins_c, "hist_c": hist_c, "peaks_c": peaks_c,
        "decisions": decisions, "raw_segments": raw_segments,
        "speech_segments": speech_segments, "detected_boundaries": boundaries,
        "rmse": rmse, "mae": mae, "errors": errors,
        "snr_db": estimate_snr_db(base["energy"], decisions),
    })
    return result


# ============================================================================
# 7. TRAINING: CHON THAM SO HISTOGRAM, KHONG DOI PIPELINE
# ============================================================================

def calibrate_training(train_files):
    """Chon bins, smoothing va W co RMSE/MAE trung binh nho nhat tren train."""
    print("\nCALIBRATION TREN TRAIN (FRAME_MS=50, EXTEND_FRAMES=5)")
    train_bases = [extract_file_base(name, TRAIN_DIR) for name in train_files]
    combinations = itertools.product(
        N_HIST_BINS_CANDIDATES, SMOOTH_WIN_CANDIDATES,
        W_ENERGY_CANDIDATES, W_CENTROID_CANDIDATES
    )
    best_score, best_params = (float("inf"), float("inf")), None

    for n_bins, smooth_win, w_energy, w_centroid in combinations:
        params = {"N_HIST_BINS": n_bins, "SMOOTH_WIN": smooth_win,
                  "W_ENERGY": w_energy, "W_CENTROID": w_centroid}
        results = [apply_pipeline(base, params) for base in train_bases]
        valid = [result for result in results if result["rmse"] is not None]
        if len(valid) != len(results):
            continue
        mean_rmse = float(np.sum([r["rmse"] for r in valid]) / len(valid))
        mean_mae = float(np.sum([r["mae"] for r in valid]) / len(valid))
        if (mean_rmse, mean_mae) < best_score:
            best_score, best_params = (mean_rmse, mean_mae), params

    if best_params is None:
        raise RuntimeError("Khong tim duoc bo tham so hop le tren tap train.")
    print(f"Best params: {best_params}")
    print(f"Train mean RMSE={best_score[0]:.2f} ms | MAE={best_score[1]:.2f} ms")
    return best_params


# ============================================================================
# 8. VE MOT FIGURE CHO MOT FILE TEST
# ============================================================================

def add_boundary_lines(axis, detected, groundtruth):
    """Ve bien detected mau xanh va ground-truth mau do tren mot truc."""
    for boundary in detected:
        axis.axvline(boundary, color="green", linewidth=1.2)
    for boundary in groundtruth:
        axis.axvline(boundary, color="red", linestyle="--", linewidth=1.2)


def position_figure(fig, index):
    """Thu sap xep 4 cua so figure vao 4 goc; bo qua neu backend khong ho tro."""
    try:
        manager = fig.canvas.manager
        width, height = 760, 500
        positions = [(0, 0), (width, 0), (0, height), (width, height)]
        x_pos, y_pos = positions[index % 4]
        if hasattr(manager.window, "wm_geometry"):
            manager.window.wm_geometry(f"{width}x{height}+{x_pos}+{y_pos}")
        elif hasattr(manager.window, "setGeometry"):
            manager.window.setGeometry(x_pos, y_pos, width, height)
    except (AttributeError, RuntimeError):
        pass


def plot_test_result(result, out_dir, figure_index):
    """Tao/lưu 1 figure gom input+STE, STE, centroid va output phan doan."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    name, total_dur = result["name"], result["total_dur"]
    time_signal, signal = result["time_signal"], result["signal"]
    times, energy = result["times"], result["energy"]

    # Plot 1: xep chong STE da chuan hoa len tin hieu vao.
    axes[0].plot(time_signal, signal, color="steelblue", linewidth=0.55,
                 label="Input waveform")
    energy_norm = energy / (float(np.max(energy)) + EPS)
    axes[0].plot(times, energy_norm, color="darkorange", linewidth=1.0,
                 label="Normalized STE")
    axes[0].set_title(f"{name}.wav - Input va STE xep chong")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", fontsize=7)

    # Plot 2-3: hai chuoi dac trung va hai nguong dong tu histogram.
    axes[1].plot(times, energy, color="darkorange", label="STE E(i)")
    axes[1].axhline(result["T_energy"], color="black", linestyle="--",
                    label=f"T1={result['T_energy']:.5g}")
    axes[1].set_title("Short-Time Energy va nguong histogram T1")
    axes[1].set_ylabel("Energy")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[2].plot(times, result["centroid"], color="purple", label="C(i)")
    axes[2].axhline(result["T_centroid"], color="black", linestyle="--",
                    label=f"T2={result['T_centroid']:.3f}")
    axes[2].set_title("Spectral Centroid va nguong histogram T2")
    axes[2].set_ylabel("DFT-bin")
    axes[2].legend(loc="upper right", fontsize=7)

    # Plot 4: output speech, bien xanh/do va sai so dinh luong.
    axes[3].plot(time_signal, signal, color="0.65", linewidth=0.55)
    for start, end in result["speech_segments"]:
        axes[3].axvspan(start, end, color="palegreen", alpha=0.45)
    add_boundary_lines(axes[3], result["detected_boundaries"],
                       result["gt_boundaries"])
    rmse_text = "N/A" if result["rmse"] is None else f"{result['rmse']:.1f} ms"
    mae_text = "N/A" if result["mae"] is None else f"{result['mae']:.1f} ms"
    snr_text = "N/A" if result["snr_db"] is None else f"{result['snr_db']:.1f} dB"
    axes[3].set_title("Output: bien xanh=algorithm, do=ground-truth | "
                      f"RMSE={rmse_text}, MAE={mae_text}, SNR~{snr_text}")
    axes[3].set_ylabel("Amplitude")
    axes[3].set_xlabel("Thoi gian (s)")

    for axis in axes:
        axis.set_xlim(0, total_dur)
        axis.grid(True, alpha=0.18)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{name}_algo2_histogram.png")
    fig.savefig(out_path, dpi=150)
    position_figure(fig, figure_index)
    return fig, out_path


# ============================================================================
# 9. MAIN - CHAY MOT LAN, DUYET 4 FILE TEST, TAO 4 FIGURE
# ============================================================================

def main():
    """Diem khoi chay duy nhat: train tham so, test 4 file, luu/hien 4 figure."""
    os.makedirs(OUT_DIR, exist_ok=True)
    train_files = find_signal_pairs(TRAIN_DIR)
    test_files = find_signal_pairs(TEST_DIR)
    if not train_files:
        raise RuntimeError(f"Khong co cap WAV/LAB trong {TRAIN_DIR}")
    if len(test_files) < 4:
        raise RuntimeError(f"Can 4 cap WAV/LAB test, chi tim thay {len(test_files)}")
    test_files = test_files[:4]

    best_params = calibrate_training(train_files)
    print("\nKET QUA TAP KIEM THU")
    test_results = []
    for index, name in enumerate(test_files):
        base = extract_file_base(name, TEST_DIR)
        result = apply_pipeline(base, best_params)
        test_results.append(result)
        _, out_path = plot_test_result(result, OUT_DIR, index)
        snr = "N/A" if result["snr_db"] is None else f"{result['snr_db']:.2f} dB"
        print(f"{name}: detected={len(result['detected_boundaries'])}, "
              f"GT={len(result['gt_boundaries'])}, RMSE={result['rmse']}, "
              f"MAE={result['mae']}, SNR~{snr}\n  Figure: {out_path}")

    # Tong hop giup so sanh thuat toan va quan sat anh huong nhieu nen.
    valid = [result for result in test_results if result["rmse"] is not None]
    if valid:
        mean_rmse = np.sum([r["rmse"] for r in valid]) / len(valid)
        mean_mae = np.sum([r["mae"] for r in valid]) / len(valid)
        print(f"\nTEST trung binh: RMSE={mean_rmse:.2f} ms | MAE={mean_mae:.2f} ms")
    print("Nhan xet SNR: SNR uoc luong thap thuong lam hai histogram chong lan, "
          "khi do nguong va bien phan doan kem on dinh hon.")
    print("Dong cac cua so figure de ket thuc chuong trinh.")
    plt.show()                   # Chi goi mot lan sau khi da tao du 4 figure.


if __name__ == "__main__":
    main()
