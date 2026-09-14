"""Chạy riêng Histogram: python main2.py; chỉ lưu ảnh: python main2.py --no-show.

Lần đầu khảo sát trên training và lưu model. Các lần sau dùng model đã lưu.
Muốn khảo sát lại: python main2.py --retrain. Không chọn tham số bằng LAB test.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from histogram_method import run_histogram_method
from common_utils import evaluate_segmentation
from common_utils import read_wav_pcm, read_lab, get_wav_lab_pairs
from common_utils import estimate_f0, estimate_snr, label_masks, add_noise

BASE = Path(__file__).resolve().parent
TRAIN = BASE / 'TinHieuHuanLuyen'
TEST = BASE / 'TinHieuKiemThu'
OUTPUT = BASE / 'KetQua_Histogram'
MODEL = OUTPUT / 'histogram_model.json'
MIN_SILENCE_MS = 300.0


def load_records(folder):
    """Nhận thư mục; đọc đủ 4 cặp WAV/LAB, trả list bản ghi tín hiệu và nhãn."""
    pairs = get_wav_lab_pairs(folder)
    wav_count = len(list(folder.glob('*.wav')))
    if len(pairs) != 4 or wav_count != 4:
        raise ValueError(f'{folder}: cần đúng 4 WAV và 4 LAB tương ứng; có {wav_count} WAV, {len(pairs)} cặp.')
    records = []
    for wav, lab in pairs:
        fs, signal = read_wav_pcm(wav)
        segments, boundaries = read_lab(lab)
        records.append(dict(wav=wav, lab=lab, fs=fs, signal=signal,
                            segments=segments, gt=boundaries))
    return records


def run_record(record, model):
    """Chạy Histogram từ WAV và tham số đã chọn; không dùng LAB để tìm ngưỡng."""
    return run_histogram_method(record['signal'], record['fs'],
        frame_ms=model['frame_ms'], W=model['W'], num_bins=model['num_bins'],
        smooth_bins=model['smooth_bins'], median_width=model['median_width'],
        min_silence_ms=MIN_SILENCE_MS)


def save_csv(path, rows):
    """Ghi list dict thành CSV UTF-8 BOM; trả None."""
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path):
    """Nhận đường dẫn; trả SHA256 để phát hiện file test sao chép training."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_model(records):
    """Khảo sát 48 cấu hình training; chọn thiếu/thừa ít nhất, rồi lỗi frame/MAE.

    Ngưỡng Energy/Centroid tự tính theo từng WAV; chỉ khóa bộ tham số chung.
    """
    from itertools import product
    rows, candidates = [], []
    for frame_ms, W, bins, smooth, median in product(
            (20.0, 50.0), (3.0, 5.0), (40, 80), (1, 3), (1, 3, 5)):
        model = dict(frame_ms=frame_ms, W=W, num_bins=bins,
                     smooth_bins=smooth, median_width=median)
        scores = []
        for record in records:
            result = run_record(record, model)
            score = evaluate_segmentation(result, record['segments'], record['gt'])
            scores.append(score)
            rows.append(dict(file=record['wav'].name, **model, **score,
                             T_energy=result['T_energy'], T_centroid=result['T_centroid'],
                             fallback=result['fallback']))
        missing = sum(s['missing'] + s['extra'] for s in scores)
        frame_error = float(np.mean([s['balanced_frame_error'] for s in scores]))
        count = sum(s['matched'] for s in scores)
        mae = sum((s['MAE_ms'] or 0)*s['matched'] for s in scores)/count if count else float('inf')
        candidates.append(((missing, frame_error, mae), model))
    # Mọi tham số được chọn trước khi đọc test. Lần sau tải lại JSON này.
    _, best = min(candidates, key=lambda item: item[0])
    best['training_wav_hashes'] = [fingerprint(r['wav']) for r in records]
    best['selection'] = 'training: unmatched boundaries, balanced frame error, matched MAE'
    best['min_silence_ms'] = MIN_SILENCE_MS
    save_csv(OUTPUT / 'training_survey.csv', rows)
    MODEL.write_text(json.dumps(best, indent=2), encoding='utf-8')
    return best


def arrange(fig, index):
    """Nhận figure/index 0..3; xếp bốn góc nếu backend Tk/Qt hỗ trợ."""
    window = getattr(fig.canvas.manager, 'window', None)
    if window is None:
        return
    if hasattr(window, 'winfo_screenwidth'):
        width = window.winfo_screenwidth() // 2
        height = (window.winfo_screenheight() - 90) // 2
        window.wm_geometry(f'{width}x{height}+{index % 2 * width}+{index // 2 * height}')
    elif hasattr(window, 'screen') and hasattr(window, 'setGeometry'):
        area = window.screen().availableGeometry()
        width, height = area.width() // 2, area.height() // 2
        window.setGeometry(area.x() + index % 2 * width,
                           area.y() + index // 2 * height, width, height)


def draw_result(record, result, score, index, training_copy, pitch_data, snr):
    """Một figure/file: waveform, hai feature, trạng thái, F0 và hai histogram."""
    fig = plt.figure(figsize=(13, 9), layout='constrained')
    grid = fig.add_gridspec(5, 3, width_ratios=[1, 1, .85], height_ratios=[1.7, 1, 1, .7, 1])
    axes = [fig.add_subplot(grid[i, :2]) for i in range(5)]
    hist_axes = [fig.add_subplot(grid[:2, 2]), fig.add_subplot(grid[2:, 2])]
    t = np.arange(len(record['signal'])) / record['fs']
    axes[0].plot(t, record['signal'], color='.5', linewidth=.5, label='Waveform')
    # Energy và waveform khác đơn vị: trục phải hiển thị STE đúng thang đo.
    energy_axis = axes[0].twinx()
    energy_axis.plot(result['frame_times'], result['energy'], color='darkorange', linewidth=.7)
    energy_axis.set_ylabel('STE', color='darkorange', fontsize=9)
    for values, color, style, label in ((record['gt'], 'red', '--', 'Ground truth'),
                                        (result['boundaries'], 'blue', '-', 'Histogram')):
        for i, boundary in enumerate(values):
            axes[0].axvline(boundary, color=color, linestyle=style,
                           label=label if i == 0 else None)
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(fontsize=8, ncol=3, loc='upper right')

    # Feature gốc mờ, sau trung vị đậm; đường ngang là ngưỡng histogram.
    for ax, key, threshold, unit in ((axes[1], 'energy', 'T_energy', 'STE'),
                                    (axes[2], 'centroid', 'T_centroid', 'Centroid (Hz)')):
        ax.plot(result['frame_times'], result[key], color='.75', label='Raw')
        ax.plot(result['frame_times'], result['smooth_'+key], color='darkorange', label='Median')
        ax.axhline(result[threshold], color='black', linestyle='--',
                   label=f"T={result[threshold]:.4g}")
        ax.set_ylabel(unit)
        ax.legend(fontsize=8, ncol=3, loc='upper right')
    for key, color, label in (('raw_states', '.65', 'Before 300 ms'),
                              ('final_states', 'blue', 'After 300 ms')):
        axes[3].step(result['frame_times'], result[key], where='mid', color=color, label=label)
    axes[3].set_yticks([0, 1], ['Silence', 'Speech'])
    axes[3].set_ylim(-.1, 1.3)
    axes[3].legend(fontsize=8, ncol=2, loc='upper right')
    axes[4].plot(pitch_data[0], pitch_data[1], color='purple', marker='.', markersize=2)
    axes[4].set_ylabel('F0 (Hz)')
    axes[4].set_ylim(60, 410)
    axes[4].set_xlabel('Time (s)')
    for ax in axes:
        ax.set_xlim(0, len(record['signal']) / record['fs'])
        ax.grid(alpha=.2)
        if ax != axes[-1]:
            ax.tick_params(labelbottom=False)

    # Vẽ đúng hai histogram đã dùng, vị trí M1/M2 và ngưỡng suy ra.
    for ax, key, threshold, unit in ((hist_axes[0], 'energy_hist', 'T_energy', 'Energy'),
                                    (hist_axes[1], 'centroid_hist', 'T_centroid', 'Centroid (Hz)')):
        hist = result[key]
        ax.plot(hist['centers'], hist['counts'], color='.75', label='Counts')
        ax.plot(hist['centers'], hist['smoothed'], color='teal', label='Smoothed')
        for peak in ('M1', 'M2'):
            if hist[peak] is not None:
                ax.axvline(hist[peak], color='purple', linestyle=':', label=peak)
        ax.axvline(result[threshold], color='black', linestyle='--', label='Threshold')
        ax.set_xlabel(unit)
        ax.set_ylabel('Frame count')
        ax.set_title('Fallback: 0.5 mean' if hist['fallback'] else 'First two local maxima', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    mae = 'N/A' if score['MAE_ms'] is None else f"{score['MAE_ms']:.1f} ms"
    rmse = 'N/A' if score['RMSE_ms'] is None else f"{score['RMSE_ms']:.1f} ms"
    source = 'TRAINING COPY - trial' if training_copy else 'TEST'
    snr_text = 'N/A' if snr is None else f'{snr:.1f} dB'
    fig.suptitle(f"{record['wav'].name} | {source} | estimated SNR: {snr_text}\n"
                 f"Matched MAE: {mae} | RMSE: {rmse} | "
                 f"missing: {score['missing']}, extra: {score['extra']}", fontsize=12)
    fig.savefig(OUTPUT / f"{record['wav'].stem}_histogram.png", dpi=160)
    arrange(fig, index)
    return fig


def noise_survey(records, model):
    """Giữ nguyên bộ tham số (ngưỡng tự tính mỗi WAV); thử AWGN 4 mức x 3 seed, lưu CSV và ảnh không mở cửa sổ.

    Dùng LAB để đo công suất/đánh giá, không truyền LAB vào bộ phân đoạn.
    """
    rows = []
    for index, record in enumerate(records):
        mask, _ = label_masks(record['signal'], record['fs'], record['segments'])
        duplicate = fingerprint(record['wav']) in model['training_wav_hashes']
        baseline = evaluate_segmentation(run_record(record, model), record['segments'], record['gt'])
        rows.append(dict(file=record['wav'].name, training_copy=duplicate,
                         condition='original', reference_snr_db=None, seed=None,
                         measured_reference_db=None, estimated_total_snr_db=estimate_snr(
                             record['signal'], record['fs'], record['segments']), **baseline))
        # Nhiễu cùng seed giữa các mức: chỉ thay công suất để so sánh có kiểm soát.
        for level in (30, 20, 10, 0):
            for trial in range(3):
                seed = 20260914 + index * 10 + trial
                noisy, measured = add_noise(record['signal'], mask, level, seed)
                modified = dict(record, signal=noisy)
                score = evaluate_segmentation(run_record(modified, model), record['segments'], record['gt'])
                rows.append(dict(file=record['wav'].name, training_copy=duplicate,
                                 condition='added_white_noise', reference_snr_db=level,
                                 seed=seed, measured_reference_db=measured,
                                 estimated_total_snr_db=estimate_snr(noisy, record['fs'], record['segments']),
                                 **score))
    save_csv(OUTPUT / 'snr_survey.csv', rows)

    # Đồ thị riêng được lưu rồi đóng: demo vẫn chỉ có bốn cửa sổ file gốc.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for record in records:
        values = [r for r in rows if r['file'] == record['wav'].name]
        x = [30, 20, 10, 0]
        for ax, metric, label in ((axes[0], 'balanced_frame_error', 'Balanced frame error (%)'),
                                  (axes[1], 'unmatched', 'Missing + extra boundaries')):
            series = [[(r['missing'] + r['extra'] if metric == 'unmatched'
                        else 100 * r[metric]) for r in values
                       if r['reference_snr_db'] == level] for level in x]
            ax.errorbar(x, np.mean(series, axis=1), yerr=np.std(series, axis=1),
                        marker='o', capsize=3, label=record['wav'].stem)
            ax.set_ylabel(label)
            ax.set_xlabel('Original speech-region power / added-noise power (dB)')
            ax.set_xticks(x)
            ax.invert_xaxis() if not ax.xaxis_inverted() else None
            ax.grid(alpha=.25)
    axes[1].legend(fontsize=9)
    copied = any(r['training_copy'] for r in rows)
    fig.suptitle('Added-noise experiment | fixed Histogram parameters; adaptive thresholds | mean +/- std, 3 trials\n'
                 + ('Includes training copies; not independent test' if copied else 'Test recordings')
                 + ' | dB is not total recording SNR', fontsize=11)
    fig.savefig(OUTPUT / 'snr_survey.png', dpi=160)
    plt.close(fig)


def main():
    """Đọc tùy chọn, học/tải model, chạy 4 file và xuất ảnh/CSV/JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrain', action='store_true', help='Khảo sát lại bằng training')
    parser.add_argument('--no-show', action='store_true', help='Lưu PNG, không mở cửa sổ')
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend('Agg')
    OUTPUT.mkdir(exist_ok=True)
    if args.retrain or not MODEL.exists():
        model = train_model(load_records(TRAIN))
    else:
        model = json.loads(MODEL.read_text(encoding='utf-8'))
    print('Histogram model:', {k: model[k] for k in ('frame_ms', 'W', 'num_bins', 'smooth_bins', 'median_width')})

    # Nếu chưa có test, vẫn chạy training và ghi rõ nguồn trên mọi kết quả.
    folder = TEST if TEST.exists() else TRAIN
    records = load_records(folder)
    rows, details = [], []
    for index, record in enumerate(records):
        result = run_record(record, model)
        score = evaluate_segmentation(result, record['segments'], record['gt'])
        duplicate = fingerprint(record['wav']) in model['training_wav_hashes']
        pitch_data = estimate_f0(record['signal'], record['fs'])
        snr = estimate_snr(record['signal'], record['fs'], record['segments'])
        valid_pitch = pitch_data[1][np.isfinite(pitch_data[1])]
        row = dict(file=record['wav'].name, training_copy=duplicate,
                   T_energy=result['T_energy'], T_centroid=result['T_centroid'],
                   fallback=result['fallback'], estimated_snr_db=snr,
                   estimated_F0mean_Hz=float(np.mean(valid_pitch)) if len(valid_pitch) else None,
                   estimated_F0std_Hz=float(np.std(valid_pitch)) if len(valid_pitch) else None,
                   **score)
        rows.append(row)
        print(row)
        print('  GT:', record['gt'].tolist(), 'Histogram:', result['boundaries'].tolist())

        # Lưu đầy đủ frame để kiểm tra chỗ mất biên mà không chạy lại.
        details.append(dict(**row, gt=record['gt'].tolist(),
                            boundaries=result['boundaries'].tolist(),
                            raw_boundaries=result['raw_boundaries'].tolist(),
                            frame_times=result['frame_times'].tolist(),
                            energy=result['energy'].tolist(), centroid=result['centroid'].tolist(),
                            raw_states=result['raw_states'].tolist(),
                            final_states=result['final_states'].tolist()))
        save_csv(OUTPUT / f"{record['wav'].stem}_f0.csv", [
            dict(time_s=float(t), F0_Hz=float(f) if np.isfinite(f) else None,
                 periodicity=float(c)) for t, f, c in zip(*pitch_data)])
        draw_result(record, result, score, index, duplicate, pitch_data, snr)
    save_csv(OUTPUT / 'histogram_metrics.csv', rows)
    (OUTPUT / 'histogram_details.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    noise_survey(records, model)
    print('Saved:', OUTPUT)
    if args.no_show:
        plt.close('all')
    else:
        plt.show()


if __name__ == '__main__':
    main()
