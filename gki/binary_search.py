"""Thuat toan hoc nguong speech/silence bang tim kiem nhi phan."""

import numpy as np


def find_threshold(silence_values, speech_values):
    """Tim nguong. Dau vao: log(MA) hai lop. Tra ve: threshold va so vong lap."""
    if silence_values.size == 0 or speech_values.size == 0:
        raise ValueError("Du lieu huan luyen phai co ca silence va speech")

    # Chi giu cac gia tri nam trong vung chong lap cua hai lop.
    lower = np.min(speech_values)
    upper = np.max(silence_values)
    if lower >= upper:
        return float((upper + lower) / 2.0), 0
    f = silence_values[(silence_values >= lower) & (silence_values <= upper)]
    g = speech_values[(speech_values >= lower) & (speech_values <= upper)]

    # Thu hep khoang den khi so phan tu hai phia nguong khong con thay doi.
    t_min, t_max = lower, upper
    old_counts = (-1, -1)
    threshold = (t_min + t_max) / 2.0
    for iteration in range(1, 101):
        threshold = (t_min + t_max) / 2.0
        counts = (int(np.sum(f < threshold)), int(np.sum(g > threshold)))
        if counts == old_counts or t_max - t_min <= 1e-12:
            return float(threshold), iteration
        old_counts = counts

        # Phuong trinh (2.8) can bang dien tich nham lan cua hai lop.
        confusion = np.mean(np.maximum(f - threshold, 0.0))
        confusion -= np.mean(np.maximum(threshold - g, 0.0))
        if confusion > 0:
            t_min = threshold
        else:
            t_max = threshold
    return float(threshold), 100
