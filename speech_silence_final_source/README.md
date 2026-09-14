# Speech/Silence Segmentation - Final Source

## Chạy thử riêng Binary (mới)

Mã nguồn chạy Binary chỉ cần 3 file:
- `main1.py`: chạy và vẽ kết quả.
- `binary_search_method.py`: thuật toán Binary và thống kê MA.
- `common_utils.py`: đọc dữ liệu, đánh giá biên, F0 và SNR dùng chung.

Các hàm phụ đã được gộp; không cần file diagnostics, speech_analysis hoặc test riêng.

Run `main1.py` trong môi trường có NumPy và Matplotlib:

```powershell
python main1.py
python main1.py --retrain --no-show
```

- Lần đầu khảo sát 8 cấu hình trên training: khung/hop 10, 20, 30, 50 ms;
  MA gốc hoặc MA sau khi trừ trung bình mỗi khung (tùy chọn tiền xử lý).
- Chọn ít biên thiếu/thừa nhất, sau đó lỗi frame cân bằng thấp nhất, rồi MAE.
  Đây là lựa chọn trong lưới thử, chưa khẳng định ngưỡng tối ưu toàn cục.
- Model lưu tại `KetQua_Binary/binary_model.json`; những lần sau tải lại,
  không tune theo test. Dùng `--retrain` khi chủ động khảo sát lại training.
- Xuất 4 PNG, `training_survey.csv`, `binary_metrics.csv`, `binary_details.json`.
- Biên được ghép một-một, đúng hướng, theo thời gian, sai lệch <= 150 ms.
  MAE/RMSE chỉ tính trên cặp ghép; luôn đọc cùng missing/extra/matched.
- WAV trùng training được phát hiện bằng SHA256 và ghi TRAINING COPY trên hình.
- `main.py` vẫn dùng cấu hình cũ; các mục “đã khóa” phía dưới là mô tả cũ,
  không phải kết quả khảo sát mới của `main1.py`.
- Đã thêm F0 tự viết bằng tương quan chuẩn hóa: frame 40 ms, hop 10 ms,
  miền 70–400 Hz, ngưỡng tuần hoàn 0.65; không dùng LAB để tạo F0.
  Đây là ước lượng đơn giản, có thể sai bội/ước tần số hoặc nhầm hữu thanh.
  Khoảng không có F0 để trống; mỗi file có thêm CSV F0 theo thời gian.
- `estimated_snr_db` dùng nhãn LAB để ước lượng công suất nền từ silence,
  lấy công suất vùng speech trừ công suất nền rồi tính tỉ lệ với nền.
  Giả định nhiễu cộng ổn định; không phải SNR chuẩn vì không có tiếng sạch.
- `snr_survey.csv`/`snr_survey.png`: thêm nhiễu trắng tại 30, 20, 10, 0 dB,
  mỗi mức 3 seed, giữ nguyên model. dB tham chiếu công suất WAV gốc vùng speech
  trên công suất nhiễu thêm, KHÔNG phải tổng SNR bản thu. CSV có cả baseline,
  SNR tổng ước lượng, biên thiếu/thừa và MAE/RMSE trên cặp ghép.
  Hình khảo sát chỉ lưu, không mở thêm cửa sổ ngoài 4 figure demo.
- Nguồn ý tưởng F0 (không dùng thư viện Praat):
  https://www.fon.hum.uva.nl/praat/manual/pitch_analysis_by_raw_autocorrelation.html

## Chạy riêng Histogram

Run `main2.py` (hoặc `python main2.py --no-show` để chỉ lưu ảnh).
Chỉ cần `main2.py`, `histogram_method.py`, `common_utils.py` và hai thư mục dữ liệu.
Dùng `--retrain` để khảo sát lại, không tune từ LAB test.

- Luồng: STE + spectral centroid -> histogram -> hai cực đại đầu theo trục feature
  -> T=(W*M1+M2)/(W+1) -> trung vị thời gian hai lượt -> AND hai điều kiện
  -> đổi khoảng lặng dưới 300 ms thành speech -> biên và đánh giá.
- Sửa đỉnh phẳng: chỉ đếm một cực đại; bỏ đỉnh rỗng. Không đủ hai cực đại thì
  dùng 0.5*mean(feature), đánh dấu fallback. Đây là quy tắc dự phòng,
  không đảm bảo phân biệt đúng tiếng nói/nhiễu với histogram một đỉnh.
- Khảo sát 48 cấu hình training: frame/hop 20 hoặc 50 ms; W=3 hoặc 5;
  bins=40 hoặc 80; smooth histogram=1 hoặc 3; median thời gian=1, 3 hoặc 5.
  Chọn theo biên thiếu/thừa, lỗi frame cân bằng rồi MAE; chỉ tối ưu trong lưới này.
  Median=1 nghĩa là bỏ làm mượt thời gian. Bộ đang chọn có median=1;
  đây là biến thể đã khảo sát, không phải khẳng định sao chép mọi chi tiết bản gốc.
- Điều chỉnh theo đề: hậu xử lý khoảng lặng tối thiểu 300 ms;
  không mở rộng biên speech thêm 250 ms vì làm thay đổi vị trí biên cần chấm.
- Khóa bộ tham số trong `KetQua_Histogram/histogram_model.json`, nhưng ngưỡng
  Energy/Centroid được tính tự động từ từng WAV mà không dùng LAB của WAV đó.
- Một lần chạy tạo 4 figure: waveform/STE, hai đặc trưng và ngưỡng, trạng thái,
  F0, hai histogram có M1/M2/T. Có SNR ước lượng và khảo sát nhiễu như Binary.
- Kết quả trong `KetQua_Histogram`: 4 PNG, model, training_survey.csv,
  histogram_metrics.csv, histogram_details.json, CSV F0 và snr_survey.csv/png.
  Các file test sao chép training được ghi rõ; chưa là kiểm thử độc lập.
- Tham khảo phương pháp Energy/Centroid của tác giả:
  https://www.mathworks.com/matlabcentral/fileexchange/28826-silence-removal-in-speech-signals
  Mô tả các bước histogram/median: https://www.mathworks.com/help/audio/ref/detectspeech.html
  Lưu ý detectSpeech hiện dùng spectral spread; bài này giữ spectral centroid theo đề.

## Cấu trúc

```text
GK/
├── main.py
├── common_utils.py
├── binary_search_method.py
├── histogram_method.py
│
├── TinHieuHuanLuyen/
│   ├── *.wav
│   └── *.lab
│
├── TinHieuKiemThu/
│   ├── 4 file *.wav
│   └── 4 file *.lab
│
└── KetQua_Test/     # tự tạo sau khi chạy
```

## Vai trò từng file

- `main.py`: file duy nhất được Run khi demo; chạy 4 test file và tạo 4 figure.
- `binary_search_method.py`: toàn bộ thuật toán Binary Search.
- `histogram_method.py`: toàn bộ thuật toán Histogram.
- `common_utils.py`: đọc WAV/LAB, framing, hậu xử lý, boundary, MAE/RMSE.

## Không dùng toolbox xử lý tín hiệu

Không dùng `scipy`, `librosa`, `pandas`, `sklearn`.

Chỉ cần:

```powershell
pip install numpy matplotlib
```

`np.fft.rfft` được dùng cho FFT vì đây là hàm built-in của NumPy.

## Chạy demo

Đứng tại thư mục `GK`:

```powershell
python main.py
```

Chỉ chạy **một lần**.

Kết quả:
- 4 figure, mỗi figure ứng với 1 test file.
- Mỗi figure có waveform + boundary cuối + MA/Energy/Centroid + threshold.
- Chương trình cố gắng xếp 4 cửa sổ ở 4 góc màn hình.
- Ảnh PNG và `test_metrics.csv` được lưu vào `KetQua_Test/`.

## Tham số đã khóa từ training

Binary Search:
- MA
- frame = 10 ms
- hop = 10 ms
- threshold học từ `TinHieuHuanLuyen`

Histogram:
- frame = 50 ms, non-overlap
- W = 3
- bins = 80
- smooth = 1
- Silence < 300 ms -> Speech
- không dùng extension 250 ms
