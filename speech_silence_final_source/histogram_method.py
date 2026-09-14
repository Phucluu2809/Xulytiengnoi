"""
histogram_method.py
Thuật toán Speech/Silence dựa trên Energy + Spectral Centroid + Histogram.
"""

import numpy as np

from common_utils import (
    frame_signal,
    remove_short_silence,
    states_to_boundaries,
)


def short_time_energy(frames):
    """
    Tính Short-Time Energy cho từng frame.

    Input:
        frames: ma trận frame.
    Return:
        energy: năng lượng trung bình từng frame.
    """
    # ------------------------------------------------------------
    # E = mean(x[n]^2). Tự tính từ mẫu tín hiệu.
    # ------------------------------------------------------------
    return np.mean(frames ** 2, axis=1)


def spectral_centroid(frames, fs):
    """
    Tính Spectral Centroid cho từng frame.

    Input:
        frames: ma trận frame.
        fs: sampling rate.
    Return:
        centroid: trọng tâm phổ theo Hz.
    """
    num_frames = frames.shape[0]
    frame_len = frames.shape[1]
    centroid = np.zeros(num_frames, dtype=np.float64)

    # ------------------------------------------------------------
    # Dùng FFT built-in NumPy; phần centroid tự tính bằng sum.
    # ------------------------------------------------------------
    freq = np.fft.rfftfreq(frame_len, d=1.0 / fs)

    for i in range(num_frames):
        spectrum = np.abs(np.fft.rfft(frames[i]))
        numerator = np.sum(freq * spectrum)
        denominator = np.sum(spectrum)

        if denominator > 1e-12:
            centroid[i] = numerator / denominator
        else:
            centroid[i] = 0.0

    return centroid


def build_histogram(values, num_bins):
    """
    Tự xây histogram 1 chiều, không dùng np.histogram.

    Input:
        values: các giá trị feature.
        num_bins: số bin.
    Return:
        counts: số mẫu từng bin.
        centers: tâm từng bin.
    """
    v_min = float(np.min(values))
    v_max = float(np.max(values))

    # ------------------------------------------------------------
    # Tránh bin_width = 0 nếu mọi giá trị giống nhau.
    # ------------------------------------------------------------
    if abs(v_max - v_min) < 1e-15:
        v_max = v_min + 1e-12

    bin_width = (v_max - v_min) / num_bins
    counts = np.zeros(num_bins, dtype=np.float64)
    centers = np.zeros(num_bins, dtype=np.float64)

    for i in range(num_bins):
        centers[i] = v_min + (i + 0.5) * bin_width

    # ------------------------------------------------------------
    # Tự xác định bin index cho từng feature value.
    # ------------------------------------------------------------
    for value in values:
        index = int((value - v_min) / bin_width)

        if index < 0:
            index = 0
        elif index >= num_bins:
            index = num_bins - 1

        counts[index] += 1.0

    return counts, centers


def smooth_histogram(counts, smooth_bins):
    """
    Làm mượt histogram bằng trung bình trượt tự viết.

    Input:
        counts: histogram gốc.
        smooth_bins: số bin trong cửa sổ trung bình.
    Return:
        smoothed: histogram sau làm mượt.
    """
    smooth_bins = int(max(1, smooth_bins))

    if smooth_bins == 1:
        return counts.copy()

    # ------------------------------------------------------------
    # Với mỗi bin, lấy mean các bin lân cận tồn tại.
    # ------------------------------------------------------------
    half = smooth_bins // 2
    smoothed = np.zeros_like(counts, dtype=np.float64)

    for i in range(len(counts)):
        left = max(0, i - half)
        right = min(len(counts), i + half + 1)
        smoothed[i] = np.mean(counts[left:right])

    return smoothed


def find_local_maxima(values):
    """
    Tự tìm local maxima, có xét cả hai endpoint.

    Input:
        values: histogram đã làm mượt.
    Return:
        peaks: index các local maxima.
    """
    peaks, i = [], 0
    # Một plateau chỉ tạo một cực đại; bỏ các plateau có số đếm bằng 0.
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j] == values[i]:
            j += 1
        left = values[i - 1] if i else -1
        right = values[j] if j < len(values) else -1
        if values[i] > 0 and values[i] > left and values[i] > right:
            peaks.append((i + j - 1) // 2)
        i = j
    return np.asarray(peaks, dtype=int)


def histogram_threshold(values, W=3.0, num_bins=80, smooth_bins=1):
    """
    Tìm threshold từ hai local maxima đầu tiên của histogram.

    Input:
        values: feature values.
        W: trọng số threshold.
        num_bins: số bin histogram.
        smooth_bins: mức làm mượt.
    Return:
        threshold và thông tin histogram.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError('Feature phải là vector hữu hạn, không rỗng.')
    if not np.isfinite(W) or W <= 0 or int(num_bins) != num_bins or num_bins < 2:
        raise ValueError('W phải dương; bins phải là số nguyên >= 2.')
    if int(smooth_bins) != smooth_bins or smooth_bins < 1 or smooth_bins % 2 == 0:
        raise ValueError('Cửa sổ làm mượt histogram phải nguyên lẻ, dương.')
    counts, centers = build_histogram(values, int(num_bins))
    smoothed = smooth_histogram(counts, smooth_bins)
    peaks = find_local_maxima(smoothed)

    # Ngoại lệ không đủ hai mode: dùng nửa mean, ghi rõ fallback trong kết quả.
    # Đây là quy tắc dự phòng, không giả định đã tìm được hai lớp phân biệt.
    if len(peaks) < 2:
        return .5 * float(np.mean(values)), {
            'counts': counts, 'centers': centers, 'smoothed': smoothed,
            'M1': None, 'M2': None, 'fallback': True,
        }

    # ------------------------------------------------------------
    # M1, M2 là local maximum thứ nhất và thứ hai.
    # T = (W*M1 + M2)/(W+1).
    # ------------------------------------------------------------
    p1 = int(peaks[0])
    p2 = int(peaks[1])

    M1 = float(centers[p1])
    M2 = float(centers[p2])
    threshold = (W * M1 + M2) / (W + 1.0)

    return threshold, {
        "counts": counts,
        "centers": centers,
        "smoothed": smoothed,
        "M1": M1,
        "M2": M2,
        "fallback": False,
    }


def median_smooth(values, width=5, passes=2):
    """Nhận feature, cửa sổ lẻ và số lượt; trả trung vị trượt tự viết.

    Đầu/cuối dùng các phần tử hiện có, không đệm zero; width=1 giữ nguyên.
    """
    if int(width) != width or width < 1 or width % 2 == 0:
        raise ValueError('Median width phải nguyên lẻ, dương.')
    output = np.asarray(values, dtype=float).copy()
    for _ in range(passes):
        previous = output.copy()
        for i in range(len(output)):
            window = sorted(previous[max(0, i-width//2):i+width//2+1])
            n = len(window)
            output[i] = (window[(n-1)//2] + window[n//2]) / 2
    return output


def run_histogram_method(
    signal,
    fs,
    frame_ms=50.0,
    W=3.0,
    num_bins=80,
    smooth_bins=1,
    min_silence_ms=300.0,
    median_width=5
):
    """
    Chạy Histogram method trên một tín hiệu.

    Input:
        signal, fs: tín hiệu và sampling rate.
        frame_ms, W, num_bins, smooth_bins: tham số đã tune.
        min_silence_ms: Silence tối thiểu.
        median_width: cửa sổ trung vị thời gian, áp dụng 2 lượt sau tìm ngưỡng.
    Return:
        dict chứa feature, threshold, states và boundaries.
    """
    frames, frame_times = frame_signal(
        signal,
        fs,
        frame_ms,
        frame_ms
    )

    # ------------------------------------------------------------
    # Tính hai feature của thuật toán Histogram.
    # ------------------------------------------------------------
    energy = short_time_energy(frames)
    centroid = spectral_centroid(frames, fs)

    # ------------------------------------------------------------
    # Mỗi file tự tìm threshold từ histogram của chính file đó.
    # ------------------------------------------------------------
    T_energy, energy_hist = histogram_threshold(
        energy,
        W,
        num_bins,
        smooth_bins
    )

    T_centroid, centroid_hist = histogram_threshold(
        centroid,
        W,
        num_bins,
        smooth_bins
    )

    # ------------------------------------------------------------
    # Speech khi cả Energy và Centroid vượt threshold.
    # ------------------------------------------------------------
    smooth_energy = median_smooth(energy, median_width)
    smooth_centroid = median_smooth(centroid, median_width)
    raw_states = (
        (smooth_energy > T_energy)
        & (smooth_centroid > T_centroid)
    ).astype(int)

    # ------------------------------------------------------------
    # Silence < 300 ms -> Speech. Không dùng extension 250 ms.
    # ------------------------------------------------------------
    final_states = remove_short_silence(
        raw_states,
        round(frame_ms * fs / 1000.0) * 1000.0 / fs,
        min_silence_ms
    )

    boundaries = states_to_boundaries(
        final_states,
        frame_times
    )

    return {
        "frame_times": frame_times,
        "energy": energy,
        "centroid": centroid,
        "T_energy": T_energy,
        "T_centroid": T_centroid,
        "energy_hist": energy_hist,
        "centroid_hist": centroid_hist,
        "raw_states": raw_states,
        "final_states": final_states,
        "boundaries": boundaries,
        "raw_boundaries": states_to_boundaries(raw_states, frame_times),
        "smooth_energy": smooth_energy,
        "smooth_centroid": smooth_centroid,
        "fallback": energy_hist['fallback'] or centroid_hist['fallback'],
    }
