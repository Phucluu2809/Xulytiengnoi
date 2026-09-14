"""
common_utils.py
Các hàm dùng chung cho cả hai thuật toán Speech/Silence.

Không dùng SciPy/librosa cho xử lý tín hiệu.
Chỉ dùng Python standard library + NumPy cho các phép toán cơ bản.
"""

from pathlib import Path
import wave
import numpy as np


def read_wav_pcm(file_path):
    """
    Đọc file WAV PCM bằng thư viện chuẩn Python.

    Input:
        file_path: đường dẫn file .wav.
    Return:
        fs: tần số lấy mẫu (Hz).
        signal: tín hiệu mono float64, chuẩn hóa gần [-1, 1].
    """
    # ------------------------------------------------------------
    # Mở WAV và đọc thông tin cơ bản của file.
    # wave là thư viện chuẩn Python, không xử lý tín hiệu.
    # ------------------------------------------------------------
    with wave.open(str(file_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        fs = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # ------------------------------------------------------------
    # Chuyển byte PCM thành số nguyên theo độ rộng mẫu.
    # WAV 24-bit cần tự ghép 3 byte thành signed integer.
    # ------------------------------------------------------------
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = data - 128.0
        scale = 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        scale = 32768.0
    elif sample_width == 3:
        bytes_3 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        value = (
            bytes_3[:, 0].astype(np.int32)
            | (bytes_3[:, 1].astype(np.int32) << 8)
            | (bytes_3[:, 2].astype(np.int32) << 16)
        )
        value = np.where(value & 0x800000, value | ~0xFFFFFF, value)
        data = value.astype(np.float64)
        scale = float(2 ** 23)
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64)
        scale = float(2 ** 31)
    else:
        raise ValueError(f"Không hỗ trợ WAV {sample_width * 8}-bit.")

    # ------------------------------------------------------------
    # Nếu WAV nhiều kênh, lấy trung bình để tạo mono.
    # Sau đó chuẩn hóa biên độ theo miền giá trị PCM.
    # ------------------------------------------------------------
    if n_channels > 1:
        data = data.reshape(-1, n_channels)
        data = np.mean(data, axis=1)

    signal = data / scale
    return fs, signal


def read_lab(file_path):
    """
    Đọc ground truth từ file .lab và gộp v/uv thành Speech.

    Input:
        file_path: đường dẫn file .lab.
    Return:
        segments: list (start, end, state), state speech/silence.
        boundaries: thời điểm chuyển Speech <-> Silence.
    """
    raw_segments = []

    # ------------------------------------------------------------
    # Đọc từng dòng. F0mean/F0std không thuộc segmentation.
    # sil -> Silence; v và uv đều được gộp thành Speech.
    # ------------------------------------------------------------
    with open(file_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].lower() in ("f0mean", "f0std"):
                continue
            if len(parts) < 3:
                continue

            start = float(parts[0])
            end = float(parts[1])
            label = parts[2].lower()
            state = "silence" if label == "sil" else "speech"
            raw_segments.append((start, end, state))

    # ------------------------------------------------------------
    # Gộp các đoạn liên tiếp có cùng trạng thái.
    # Việc này loại boundary v <-> uv vì cả hai đều là Speech.
    # ------------------------------------------------------------
    segments = []
    for start, end, state in raw_segments:
        if (
            segments
            and segments[-1][2] == state
            and abs(segments[-1][1] - start) < 1e-6
        ):
            segments[-1] = (segments[-1][0], end, state)
        else:
            segments.append((start, end, state))

    # ------------------------------------------------------------
    # Boundary chuẩn chỉ xuất hiện khi trạng thái thay đổi.
    # ------------------------------------------------------------
    boundaries = []
    for i in range(len(segments) - 1):
        if segments[i][2] != segments[i + 1][2]:
            boundaries.append(segments[i][1])

    return segments, np.asarray(boundaries, dtype=float)


def get_wav_lab_pairs(folder):
    """
    Tìm các cặp WAV/LAB cùng tên trong một thư mục.

    Input:
        folder: thư mục dữ liệu.
    Return:
        list các tuple (wav_path, lab_path), sắp xếp theo tên.
    """
    folder = Path(folder)
    pairs = []

    # ------------------------------------------------------------
    # Chỉ nhận WAV khi có LAB cùng tên.
    # ------------------------------------------------------------
    for wav_path in sorted(folder.glob("*.wav")):
        lab_path = wav_path.with_suffix(".lab")
        if lab_path.exists():
            pairs.append((wav_path, lab_path))

    return pairs


def frame_signal(signal, fs, frame_ms, hop_ms):
    """
    Chia tín hiệu thành các frame ngắn bằng code tự viết.

    Input:
        signal: tín hiệu mono.
        fs: sampling rate.
        frame_ms: độ dài frame (ms).
        hop_ms: bước dịch frame (ms).
    Return:
        frames: ma trận frame.
        times: thời gian tâm từng frame.
    """
    if fs <= 0 or frame_ms <= 0 or hop_ms <= 0:
        raise ValueError("fs, frame_ms và hop_ms phải dương.")
    frame_len = int(round(frame_ms * fs / 1000.0))
    hop_len = int(round(hop_ms * fs / 1000.0))
    if frame_len < 1 or hop_len < 1:
        raise ValueError("Frame và hop phải có ít nhất một mẫu.")

    # ------------------------------------------------------------
    # Tự xác định vị trí bắt đầu của từng frame.
    # Không dùng hàm framing từ toolbox xử lý tiếng nói.
    # ------------------------------------------------------------
    starts = []
    start = 0
    while start + frame_len <= len(signal):
        starts.append(start)
        start += hop_len

    if not starts:
        raise ValueError("Tín hiệu ngắn hơn một frame.")

    # ------------------------------------------------------------
    # Cắt trực tiếp từng đoạn tín hiệu để tạo ma trận frame.
    # ------------------------------------------------------------
    frames = np.zeros((len(starts), frame_len), dtype=np.float64)
    times = np.zeros(len(starts), dtype=np.float64)

    for i, start in enumerate(starts):
        frames[i, :] = signal[start:start + frame_len]
        times[i] = (start + frame_len / 2.0) / fs

    return frames, times


def remove_short_silence(states, step_ms, min_silence_ms=300.0):
    """
    Loại các đoạn Silence ngắn hơn thời lượng tối thiểu.

    Input:
        states: 0=Silence, 1=Speech theo frame.
        step_ms: thời gian giữa hai quyết định liên tiếp.
        min_silence_ms: Silence tối thiểu.
    Return:
        states sau hậu xử lý.
    """
    output = np.asarray(states, dtype=int).copy()
    i = 0

    # ------------------------------------------------------------
    # Quét từng run Silence. Nếu ngắn hơn 300 ms thì đổi thành Speech.
    # ------------------------------------------------------------
    while i < len(output):
        if output[i] != 0:
            i += 1
            continue

        j = i
        while j < len(output) and output[j] == 0:
            j += 1

        duration_ms = (j - i) * step_ms
        if duration_ms < min_silence_ms:
            output[i:j] = 1

        i = j

    return output


def states_to_boundaries(states, frame_times):
    """
    Chuyển chuỗi state thành danh sách boundary.

    Input:
        states: 0/1 theo frame.
        frame_times: thời gian tâm frame.
    Return:
        boundaries: thời điểm state thay đổi.
    """
    boundaries = []

    # ------------------------------------------------------------
    # Khi hai frame liên tiếp khác trạng thái, boundary đặt ở trung điểm.
    # ------------------------------------------------------------
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            boundary = (frame_times[i - 1] + frame_times[i]) / 2.0
            boundaries.append(boundary)

    return np.asarray(boundaries, dtype=float)


def boundary_metrics(ground_truth, prediction):
    """
    Tính MAE và RMSE của boundary theo millisecond.

    Input:
        ground_truth: boundary chuẩn.
        prediction: boundary dự đoán.
    Return:
        mae_ms, rmse_ms, valid.
    """
    gt = np.asarray(ground_truth, dtype=float)
    pred = np.asarray(prediction, dtype=float)

    # ------------------------------------------------------------
    # Không ép ghép nếu số boundary khác nhau.
    # ------------------------------------------------------------
    if len(gt) != len(pred) or len(gt) == 0:
        return np.nan, np.nan, False

    error_ms = (pred - gt) * 1000.0

    # ------------------------------------------------------------
    # MAE = sai số tuyệt đối trung bình; RMSE phạt lỗi lớn mạnh hơn.
    # ------------------------------------------------------------
    mae_ms = float(np.mean(np.abs(error_ms)))
    rmse_ms = float(np.sqrt(np.mean(error_ms ** 2)))

    return mae_ms, rmse_ms, True


# Nhãn frame, đánh giá, F0 và SNR dùng chung cho hai thuật toán.
def make_frame_labels(frame_times, segments):
    """
    Gán ground-truth label cho từng frame training.

    Input:
        frame_times: thời gian tâm frame.
        segments: ground-truth segments từ LAB.
    Return:
        labels: 0=Silence, 1=Speech, -1=ngoài miền nhãn.
    """
    labels = np.full(len(frame_times), -1, dtype=int)

    # ------------------------------------------------------------
    # Tâm frame thuộc segment nào thì nhận state của segment đó.
    # ------------------------------------------------------------
    for i, t in enumerate(frame_times):
        for start, end, state in segments:
            if start <= t < end:
                labels[i] = 0 if state == "silence" else 1
                break

    return labels


def evaluate_segmentation(result, segments, ground_truth, tolerance_ms=150.0):
    """Nhận kết quả/LAB; trả sai số biên ghép, biên thiếu/thừa và lỗi frame.

    Ghép một-một, giữ thứ tự, cùng hướng chuyển trạng thái, lệch <= tolerance.
    Tối đa số cặp trước, tối thiểu tổng sai số tuyệt đối sau. RMSE chỉ trên cặp.
    """
    gt = np.asarray(ground_truth, dtype=float)
    pred = result['boundaries']
    changes = np.flatnonzero(np.diff(result['final_states']) != 0) + 1
    pred_types = result['final_states'][changes]
    gt_types = [int(segments[i + 1][2] == 'speech')
                for i in range(len(segments) - 1)
                if segments[i][2] != segments[i + 1][2]]

    # Quy hoạch động: mỗi ô lưu số cặp, tổng lỗi và danh sách cặp.
    n, m = len(gt), len(pred)
    table = [[(0, 0.0, []) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [table[i - 1][j], table[i][j - 1]]
            error = abs(gt[i - 1] - pred[j - 1]) * 1000.0
            if gt_types[i - 1] == pred_types[j - 1] and error <= tolerance_ms:
                count, cost, pairs = table[i - 1][j - 1]
                candidates.append((count + 1, cost + error, pairs + [(i - 1, j - 1)]))
            table[i][j] = max(candidates, key=lambda item: (item[0], -item[1]))

    # Không tạo RMSE giả khi không có cặp; vẫn báo rõ thiếu và thừa.
    count, _, pairs = table[n][m]
    errors = np.asarray([(pred[j] - gt[i]) * 1000.0 for i, j in pairs])
    labels = make_frame_labels(result['frame_times'], segments)
    class_errors = []
    for state in (0, 1):
        mask = labels == state
        if np.any(mask):
            class_errors.append(float(np.mean(result['final_states'][mask] != state)))
    return {
        'gt_count': n, 'pred_count': m, 'matched': count,
        'missing': n - count, 'extra': m - count,
        'MAE_ms': float(np.mean(np.abs(errors))) if count else None,
        'RMSE_ms': float(np.sqrt(np.mean(errors ** 2))) if count else None,
        'balanced_frame_error': float(np.mean(class_errors)) if class_errors else None,
        'match_tolerance_ms': tolerance_ms,
    }


def estimate_f0(signal, fs, frame_ms=40.0, hop_ms=10.0,
                fmin=70.0, fmax=400.0, periodicity=0.65):
    """Nhận WAV mono/fs; trả times, F0 Hz (NaN nếu vô thanh), độ tuần hoàn.

    Không dùng LAB hay biên Binary. Cổng năng lượng tương đối 1% RMS cực đại.
    Chọn đỉnh tương quan đầu tiên đạt >=95% đỉnh tốt nhất để giảm lỗi chia đôi F0.
    """
    if not 0 < fmin < fmax < fs / 2 or not 0 < periodicity <= 1:
        raise ValueError('Miền F0 hoặc ngưỡng tuần hoàn không hợp lệ.')
    frames, times = frame_signal(signal, fs, frame_ms, hop_ms)
    frames = frames - np.mean(frames, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    low, high = max(1, int(np.ceil(fs / fmax))), int(np.floor(fs / fmin))
    if frames.shape[1] < 2 * high:
        raise ValueError('Frame F0 phải chứa ít nhất hai chu kỳ thấp nhất.')
    pitch = np.full(len(frames), np.nan)
    strength = np.zeros(len(frames))

    # Tương quan hai đoạn dịch nhau; tự tính bằng tổng tích mẫu.
    for i, frame in enumerate(frames):
        if rms[i] <= max(1e-10, .01 * float(np.max(rms))):
            continue
        corr = np.zeros(high + 2)
        for lag in range(max(1, low - 1), high + 2):
            left, right = frame[:-lag], frame[lag:]
            denominator = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2))
            corr[lag] = np.sum(left * right) / denominator if denominator > 1e-20 else 0
        peaks = [k for k in range(low, high + 1)
                 if corr[k] >= corr[k - 1] and corr[k] > corr[k + 1]]
        if not peaks:
            continue
        best = max(corr[k] for k in peaks)
        strength[i] = best
        if best < periodicity:
            continue

        # Nội suy parabol quanh đỉnh giúp tránh lượng tử hóa theo lag nguyên.
        lag = next(k for k in peaks if corr[k] >= max(periodicity, .95 * best))
        denominator = corr[lag - 1] - 2 * corr[lag] + corr[lag + 1]
        offset = .5 * (corr[lag - 1] - corr[lag + 1]) / denominator if abs(denominator) > 1e-12 else 0
        value = fs / (lag + float(np.clip(offset, -.5, .5)))
        if fmin <= value <= fmax:
            pitch[i] = value
    return times, pitch, strength


def label_masks(signal, fs, segments):
    """Nhận tín hiệu/fs/LAB đã gộp; trả mask mẫu speech và silence để đánh giá."""
    speech = np.zeros(len(signal), dtype=bool)
    silence = np.zeros(len(signal), dtype=bool)
    for start, end, state in segments:
        a = max(0, int(np.ceil(start * fs)))
        b = min(len(signal), int(np.ceil(end * fs)))
        (silence if state == 'silence' else speech)[a:b] = True
    return speech, silence


def estimate_snr(signal, fs, segments):
    """Ước lượng SNR từ LAB: (P_speech_observed - P_silence)/P_silence.

    Giả định nhiễu cộng, không tương quan và công suất nền ổn định.
    Không có tiếng sạch nên đây không phải SNR ground truth; None nếu không xác định.
    """
    speech, silence = label_masks(signal, fs, segments)
    if not np.any(speech) or not np.any(silence):
        return None
    centered = signal - np.mean(signal)
    noise_power = float(np.mean(centered[silence] ** 2))
    observed_power = float(np.mean(centered[speech] ** 2))
    if noise_power <= 1e-20 or observed_power <= noise_power:
        return None
    return float(10 * np.log10((observed_power - noise_power) / noise_power))


def add_noise(signal, speech_mask, reference_snr_db, seed):
    """Trả WAV+nhiễu, tỉ lệ đo được; dB tham chiếu công suất WAV gốc vùng speech.

    WAV gốc đã có nhiễu: tỉ lệ này KHÔNG phải tổng SNR cuối. Không clip/chuẩn hóa.
    """
    if not np.any(speech_mask):
        raise ValueError('Cần vùng speech để chuẩn công suất nhiễu.')
    power = float(np.mean(signal[speech_mask] ** 2))
    if power <= 0:
        raise ValueError('Công suất tham chiếu phải dương.')
    noise = np.random.default_rng(seed).standard_normal(len(signal))
    noise -= np.mean(noise)
    noise *= np.sqrt(power / (10 ** (reference_snr_db / 10) * np.mean(noise[speech_mask] ** 2)))
    measured = float(10 * np.log10(power / np.mean(noise[speech_mask] ** 2)))
    return signal + noise, measured
