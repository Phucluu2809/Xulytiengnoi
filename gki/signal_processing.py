"""Cac ham doc du lieu va xu ly tin hieu dung chung cho hai thuat toan."""

from pathlib import Path
import wave

import numpy as np


FRAME_MS = 20.0
HOP_MS = 10.0
MIN_SILENCE_MS = 300.0


def read_wav(path):
    """Doc WAV PCM. Dau vao: path. Tra ve: tin hieu mono x va tan so fs."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        fs = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    # Tu giai ma cac kieu PCM thong dung, khong dung toolbox am thanh.
    if width == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(float) - 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(float)
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        value = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
        value |= b[:, 2].astype(np.int32) << 16
        x = ((value ^ 0x800000) - 0x800000).astype(float)
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(float)
    else:
        raise ValueError(f"Khong ho tro WAV {8 * width}-bit: {path}")

    # Chuyen nhieu kenh ve mono va chuan hoa bien do vao [-1, 1].
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    peak = np.max(np.abs(x)) if x.size else 0.0
    return (x / peak if peak > 0 else x), fs


def read_lab(path):
    """Doc LAB. Dau vao: path. Tra ve: cac bo (bat_dau, ket_thuc, nhan)."""
    segments = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    segments.append((float(parts[0]), float(parts[1]), parts[2]))
                except ValueError:
                    pass  # Bo qua cac dong F0mean va F0std.
    return segments


def calculate_log_ma(x, fs, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """Tinh log(MA). Dau vao: x, fs, do dai/buoc khung. Tra ve: times, feature."""
    frame_length = max(1, round(frame_ms * fs / 1000.0))
    hop_length = max(1, round(hop_ms * fs / 1000.0))
    if x.size == 0:
        return np.array([]), np.array([])

    # Tu tinh MA tren tung cua so chu nhat roi lay log de nen mien gia tri.
    starts = np.arange(0, x.size, hop_length)
    feature = np.empty(starts.size, dtype=float)
    for index, start in enumerate(starts):
        frame = x[start:min(start + frame_length, x.size)]
        feature[index] = np.log(np.mean(np.abs(frame)) + np.finfo(float).eps)
    times = np.minimum(starts + frame_length / 2.0, x.size - 1) / fs
    return times, feature


def labels_at_times(times, segments):
    """Gan nhan khung. Dau vao: times, segments. Tra ve: 0=silence, 1=speech."""
    labels = np.zeros(times.size, dtype=np.uint8)
    for left, right, label in segments:
        if label in ("v", "uv"):
            labels[(times >= left) & (times < right)] = 1
    return labels


def remove_short_silences(labels, hop_ms=HOP_MS, min_ms=MIN_SILENCE_MS):
    """Loai silence ao. Dau vao: labels, hop_ms, min_ms. Tra ve: nhan da sua."""
    result = labels.copy()
    minimum_frames = int(np.ceil(min_ms / hop_ms))
    index = 0

    # Chi xoa silence ngan nam giua hai doan speech (khoang lang ao).
    # Silence o dau/cuoi ban ghi duoc giu lai de khong lam mat bien endpoint.
    while index < result.size:
        if result[index] != 0:
            index += 1
            continue
        end = index
        while end < result.size and result[end] == 0:
            end += 1
        is_internal = index > 0 and end < result.size
        if is_internal and end - index < minimum_frames:
            result[index:end] = 1
        index = end
    return result
