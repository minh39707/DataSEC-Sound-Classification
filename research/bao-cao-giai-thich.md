# Báo cáo giải thích dự án DataSEC (để đọc hiểu)

Viết cho: bạn, người cần hiểu toàn bộ dự án trước khi thuyết trình và bảo vệ (viva). Mọi con số lấy từ code trong
`DataSEC-Sound-Classification-main/` (bảng ở `results/tables/`). Phần nào là suy đoán, tôi ghi rõ là "giả thuyết".

---

## 1. Dự án làm gì, trong 5 câu

1. **Bài toán:** nghe một đoạn âm thanh ngắn và đoán nó thuộc 1 trong 22 loại (tiếng chó sủa, còi xe, máy bay phản lực, tiếng nói, ...).
2. **Dữ liệu:** DataSEC, 5,048 file wav (23.7 giờ), mất cân bằng rất nặng: Voices 1,900 clip, Cat fights chỉ 50 clip.
3. **Cách làm:** biến âm thanh thành "ảnh" log-Mel (64 × 216) rồi đưa vào mạng nơ-ron. Ba mạng: small CNN, MobileNetV3-Small (pretrained), CRNN.
4. **Midterm:** huấn luyện ba mạng trên dữ liệu sạch, đo độ chính xác, phân tích lỗi.
5. **Final (RQ1):** làm bẩn tập test (thêm nhiễu, đổi âm lượng, che thời gian, che dải tần) rồi xem mạng tệ đi bao nhiêu.

## 2. Các khái niệm bạn cần nắm (hay bị hỏi khi bảo vệ)

**Waveform → spectrogram → log-Mel.**
- *Waveform:* áp suất không khí theo thời gian. Chỉ cho biết khi nào to/nhỏ, không đọc được tần số.
- *Spectrogram tuyến tính:* cắt tín hiệu thành khung ngắn (1024 mẫu, bước 512 ≈ 23 ms), mỗi khung biến đổi Fourier, lấy bình phương độ lớn. Trục tần số chia đều (513 bin, ~21.5 Hz mỗi bin).
- *log-Mel:* gom 513 bin thành 64 dải theo thang Mel (dải thấp hẹp, dải cao rộng, giống tai người), rồi lấy log (nén dải động; **đổi âm lượng thành phép cộng hằng số**). Đây là thứ mạng nhìn thấy.
- Lý do chọn 22.05 kHz: nhiều lớp (Glass breaking) có năng lượng trên 8 kHz; hạ xuống 16 kHz sẽ mất bớt. Đồng thời xoá được "vết" Voices/Music vốn bị cắt cứng ở 11.025 kHz.

**Macro-F1 vs accuracy.** Đoán "Voices" cho mọi clip đã được accuracy 0.376 nhưng macro-F1 chỉ 0.025. Macro-F1 coi 22 lớp ngang nhau nên là thước đo chính.

**Split.** 70/15/15, chia theo 40 lớp con, cố định trong `splits/`. Tập test 758 clip. Không bao giờ chỉnh mô hình theo tập test (chọn checkpoint bằng macro-F1 trên validation).

**Ba mạng.**
| mạng | ý tưởng | tham số | MACs |
|---|---|---:|---:|
| small_cnn | 4 khối conv, nhỏ nhất | 0.39 M | 198 M |
| mobilenetv3s | mạng ảnh pretrained ImageNet, nhân kênh xám ra 3 kênh | 1.54 M | 18 M |
| crnn | conv rồi GRU hai chiều (nhớ chuỗi theo thời gian) | 0.87 M | 487 M |

**MACs vs latency.** MACs là số phép nhân-cộng (lý thuyết). Latency là thời gian thật. MobileNet ít MACs nhất nhưng chậm nhất trên GPU (6.9 ms) vì tích chập depthwise không tận dụng tốt GPU ở batch 1.

## 3. Đối chiếu với đề midterm (có khớp không?)

| Mục đề | Trạng thái | Ghi chú |
|---|---|---|
| A. Số clip, phân bố 22 lớp | Đạt | 5,048 clip. **Đề ghi 4,292** nhưng bản Zenodo ghim DOI có 5,048; repo đã ghi rõ chênh lệch |
| A. Thống kê thời lượng (min/max/mean/median) | Đạt | `eda_duration_stats.csv` |
| A. Waveform + spectrogram ≥ 3 lớp tương phản | Đạt | 4 lớp: Gunshot, Sirens, Jet, Voices |
| A. log-Mel 2D, cân bằng lớp, lớp phổ giống nhau | Đạt | Jet–Propeller cosine 0.992 |
| A. Giải thích toán/vật lý 3 biểu diễn | Đạt | `dataset_analysis.md` A7 |
| B. Pipeline chạy lúc chạy, không lưu ảnh spectrogram | Đạt | cache chỉ chứa audio int16 |
| C. Ba mô hình yêu cầu | Đạt | small CNN, MobileNetV3-S pretrained, CRNN |
| D. Acc, macro-F1, F1 từng lớp, confusion matrix | Đạt | |
| D. Số tham số, thời gian train, MACs, latency | Đạt | |
| E1–E4 | Đạt, **nhưng kết luận phải sửa** | xem mục 4 |
| F. Notebook chạy được | Có, **chưa chạy lại** | 4 notebook sinh từ file .py; tôi chưa chạy lại chúng sau khi huấn luyện lại |
| F. Kiểm tra lỗi bằng nghe thử (qualitative audio error audit) | **Thiếu** | bảng E2 có 10 clip nhưng 2 cột "nghe thấy gì" và "nguyên nhân" còn trống. Tôi không nghe được audio. **Bạn phải tự nghe và điền** |
| F. Slide | Làm ở mục 7 | `outputs/midterm_slides.pptx` và `research/presentation-script.md` |
| F. Đề xuất câu hỏi nghiên cứu cho Final | Đạt | `docs/final_rq_proposal.md` (RQ1) |

## 4. Điều quan trọng: kết quả midterm gốc không lặp lại được

Midterm gốc chạy **1 seed trên Colab**. Tôi chạy thêm seed 43, 44 và chạy lại seed 42 trên máy bạn (RTX 2060 SUPER),
vì checkpoint seed 42 chỉ có trên Colab và phần Final cần checkpoint. Kết quả test macro-F1:

| mô hình | Colab seed 42 (midterm gốc) | local, 3 seed (mean ± std) |
|---|---:|---:|
| small_cnn | 0.884 | 0.876 ± 0.015 |
| mobilenetv3s | 0.853 | **0.901 ± 0.027** |
| crnn | **0.815** | 0.875 ± 0.011 |

- Thứ hạng cũ (small_cnn > mobilenet > crnn) **sai** khi lặp lại. Ba mô hình gần như ngang nhau; độ lệch giữa các seed (đến 0.027) bằng khoảng cách giữa mô hình.
- Nguyên nhân chính: lần Colab CRNN dừng sớm ở epoch 20 (tốt nhất epoch 12); chạy local CRNN train 40–50 epoch. Đây là chênh lệch quá trình huấn luyện, không phải CRNN kém.
- Số Colab cũ được giữ nguyên trong `results/colab_seed42/`, không trộn vào bảng nào.

**Câu trả lời E1–E4 sau 3 seed** (`midterm_ms_*.csv`):
- **E1 (lớp tốt/tệ nhất):** tốt nhất Voices (1.00), Chicken coop, Wind turbine, Music, Cicadas (≥ 0.97). Tệ nhất Jet (0.71), Train (0.73), Vehicle idling (0.74), Propeller và Vacuum/fan (0.76). Ổn định giữa các seed (std 0.06–0.09 với lớp khó). Lưu ý: Voices/Music/Wind turbine tốt một phần nhờ "dấu vết" xử lý dữ liệu (cắt ở 11 kHz, độ dài cố định).
- **E2 (nhầm lẫn nặng nhất):** Jet↔Propeller (3.2 clip/lần chạy), Bells↔Music (2.7, chủ yếu Music→Bells), Lawn mower↔Vehicle idling (2.6), Jet↔Vehicle pass-by (2.4), Vacuum↔Workshop (2.4). Toàn bộ là nhóm tiếng máy/động cơ liên tục.
- **E3 (GRU của CRNN có giúp không?):** chênh lệch CRNN − CNN theo độ dài clip: ngắn −0.008 ± 0.028, vừa +0.021 ± 0.040, dài +0.017 ± 0.019. **Không có khác biệt nhất quán**; kết luận "CRNN kém hơn trên clip dài" của midterm gốc không còn đúng. Thêm: CRNN và CNN khác cả phần conv nên không quy được cho riêng GRU.
- **E4 (cân bằng hiệu năng/chi phí):** mobilenetv3s có F1 sạch cao nhất và ít MACs nhất (18 M) nên tối ưu Pareto theo MACs; small_cnn nhanh nhất (1.18 ms) và ít tham số nhất. Không có "người thắng" duy nhất; nói rõ tiêu chí chi phí bạn chọn.

## 5. Kết quả Final: robustness (RQ1)

**Thiết lập.** Không train lại. 3 mô hình × 3 seed × (sạch + 12 điều kiện làm bẩn). Bốn kiểu làm bẩn, mỗi kiểu 3 mức:

| kiểu | mức | ý nghĩa |
|---|---|---|
| noise | SNR +20/+10/0 dB | cộng nhiễu hồng (1/f); 0 dB = nhiễu to ngang tín hiệu |
| gain | −12/−6/+6 dB | nhân âm lượng |
| tmask | 10/25/50 % | xoá (đặt 0) một đoạn liên tục của cửa sổ 5 s |
| fmask | 8/16/32 trong 64 dải | xoá một khối dải Mel |

Mỗi clip bị làm bẩn theo seed cố định theo (sample_id, kiểu, mức) nên mọi mô hình thấy đúng cùng một đầu vào bẩn (so sánh ghép cặp).

**Kết quả ở mức nặng nhất (macro-F1, sạch ≈ 0.88–0.90):**

| | small_cnn | mobilenetv3s | crnn |
|---|---:|---:|---:|
| noise 0 dB | 0.14 | 0.25 | 0.21 |
| gain +6 dB | 0.87 | 0.88 | 0.88 |
| tmask 50 % | 0.52 | 0.45 | **0.60** |
| fmask 32 dải | 0.17 | 0.16 | 0.14 |

**Đọc nhanh:**
1. **Âm lượng gần như không ảnh hưởng** (tối đa mất 0.056). Mạng không dựa vào độ lớn tuyệt đối trong khoảng ±12 dB.
2. **Nhiễu và che dải tần cực kỳ phá hoại:** nhiễu +20 dB đã mất 0.15–0.22; che chỉ 8/64 dải đã mất 0.46–0.60. Cảnh báo: che dải bằng "0 tuyệt đối" là kiểu lỗ hổng mà mạng chưa từng thấy, không giống nhiễu thực tế, nên đừng nói đây là tỷ lệ lỗi ngoài đời.
3. **Che thời gian phân biệt được các mạng:** CRNN chịu tốt nhất, MobileNet tệ nhất, khoảng tin cậy bootstrap 95 % loại trừ 0. Nhưng vì CRNN còn khác phần conv và độ dài train, **không được kết luận GRU là nguyên nhân**.
4. **Nhiễu làm mạng đoán sang lớp "động cơ":** ví dụ 149/285 clip Voices bị đoán thành Train. Giả thuyết (chưa kiểm chứng): nhiễu hồng là năng lượng dải thấp liên tục, giống phổ của Train/Vehicle idling.

**Giả thuyết đặt ra trước khi thí nghiệm (H1–H5): 4/5 không được ủng hộ.** Ví dụ H1 dự đoán nhiễu hại nhóm động cơ nhất, nhưng thực tế nhiễu hại các lớp điểm cao (Voices, Chicken coop, Wind turbine) nhất, còn nhóm động cơ là nơi lỗi "rơi vào". Báo cáo Final ghi đầy đủ; đừng bào chữa, hãy nói thẳng "giả thuyết sai" vì đó là kết quả khoa học hợp lệ.

**Mở rộng RQ4 (Pareto):** F1 trung bình trên 12 điều kiện bẩn: small_cnn 0.545, mobilenetv3s 0.586, crnn 0.564. Chưa đo VRAM tối đa (đề yêu cầu); phần này còn thiếu.

## 6. Hạn chế bạn phải nói thật khi bảo vệ

- Mô hình chỉ train trên dữ liệu sạch, không augmentation. Chưa chạy SpecAugment (RQ2) nên chưa biết sửa được bao nhiêu.
- Nhiễu là nhiễu hồng tổng hợp, không phải âm thanh đường phố/mưa thật. Kiểu "trộn nền môi trường" của đề chưa làm.
- 3 seed là ít; độ lệch giữa seed bằng độ lệch giữa mô hình ở tập sạch.
- Nhiều lớp chỉ có 8–17 clip test; thêm hoặc bớt 1 clip làm recall đổi 6–12 điểm.
- 2 clip test trùng một phần audio với train (ảnh hưởng ≤ 0.0017 macro-F1).
- Midterm chạy Colab T4, Final chạy RTX 2060 SUPER; latency hai nơi không so được.
- Chưa có nghe thử cho các clip bị lật đúng→sai; mọi giải thích âm học là giả thuyết.
- Một điểm số test lúc huấn luyện (mobilenetv3s seed 42) lệch 0.0002 so với lần suy luận lại (kernel GPU không tất định); tôi đã cập nhật log và ghi chú ngay trong file log.

## 7. Việc tôi đã sửa/thêm trong repo

- `src/perturb.py` (mới), `scripts/run_robustness.py`, `scripts/analyze_robustness.py`, `scripts/pareto_rq4.py`, `scripts/midterm_multiseed.py`, mục `robustness` trong `configs/default.yaml`, tham số `power_hook` trong `src/features.py`.
- `docs/final_report.md` (báo cáo Final dạng paper, tiếng Anh).
- Thay đổi so với đề xuất ban đầu: SNR tính theo đoạn có tiếng của clip, không tính phần zero-padding (nếu tính cả padding thì clip ngắn bị nhiễu nhẹ hơn mức danh nghĩa).
- Môi trường: junction `data/raw/DATASEC` trỏ tới `C:\ATI\DATASEC`, không sao chép dữ liệu.

## 8. Việc còn lại phải do bạn làm

1. **Nghe thử** 10 clip trong bảng E2 của `docs/midterm_answers.md` và vài clip trong `results/tables/rq1_flips.csv`, điền cột "nghe thấy gì" và "nguyên nhân".
2. Chạy lại 4 notebook (`notebooks/`) nếu muốn nộp notebook đã chạy với số mới.
3. Cập nhật `docs/midterm_answers.md` (vẫn là số Colab một seed) theo mục 4 hoặc ghi chú rằng số 3-seed nằm ở `midterm_ms_*.csv`.
4. Xuất báo cáo Final ra PDF (máy chưa có pandoc; mở `docs/final_report.md` bằng VS Code/Word rồi xuất).
5. Quyết định có chạy mở rộng SpecAugment (RQ2) không. Tôi chưa làm vì đề nói chỉ làm sau khi RQ1 đã xong, và nó cần train lại (khoảng 10 phút mỗi model-seed).
