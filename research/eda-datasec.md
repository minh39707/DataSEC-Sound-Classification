# Phân tích dataset DataSEC (giữa kỳ, phần A)

Mọi số liệu, bảng và hình trong file này đều sinh ra từ notebook `notebooks/01_eda_datasec.ipynb`, chạy trên bản `DATASEC/` trong máy. Script `scripts/eda_datasec.py` chứa cùng phân tích dưới dạng dòng lệnh và cho kết quả giống hệt. Hình nằm ở `outputs/eda/figures/`, bảng ở `outputs/eda/tables/`. Cách chạy lại ở mục 9.

## Tóm tắt các phát hiện chính

1. **Đúng phiên bản đã chốt, nhưng số clip khác với đề.** Dữ liệu trong máy chính là archive `DATASEC.zip` của Zenodo `10.5281/zenodo.17033970`: MD5 khớp, và cả 5.048 file đều khớp CRC32. Archive này có **5.048 clip (23,7 giờ)**, trong khi đề, bài báo và phần mô tả trên Zenodo đều ghi 4.292. Chênh lệch nằm trọn ở lớp **Music**: 1.001 clip thay vì 245. Cả 39 dòng còn lại của Table 2 trong bài báo đều khớp với dữ liệu. Dự án dùng archive thực tế và ghi chú sự khác biệt này trong báo cáo.
2. **Mất cân bằng nặng.** Voices (37,6%) và Music (19,8%) chiếm 57,5% số clip. Tỉ lệ lớn nhất/nhỏ nhất là 38:1, và số lớp hiệu dụng chỉ khoảng 9,8 trên 22. Bỏ hai lớp này đi thì tỉ lệ còn 4,7:1. Vì vậy Macro-F1 phải là metric chính.
3. **Thời lượng rất lệch.** Clip dài từ 0,2 s đến 481,5 s, median 8,7 s. Có 34% số clip ngắn hơn 4 s, phần lớn thuộc Voices; ngoài Voices và Music vẫn còn 17% clip dưới 4 s. Gunshot (98,6% dưới 4 s), Horn, Dog và Chicken coop đều có nhiều clip ngắn. Nếu đệm bằng số 0, độ dài đoạn im lặng sẽ thành một dấu hiệu gián tiếp để đoán lớp.
4. **Có trùng lặp và chồng lấn nội dung**, tạo nguy cơ rò rỉ giữa train và test:
   - 16 cặp file giống hệt nhau (14 cặp trong Music);
   - 5 cặp mà file ngắn là prefix chính xác của file dài;
   - 3 cặp có vài giây đầu giống hệt nhau.

   Tất cả gồm 24 nhóm, liệt kê trong `leakage_groups.csv`. Mỗi nhóm phải nằm trọn trong một split.
5. **Có dấu vết của khâu tuyển chọn dữ liệu, mô hình có thể học "đường tắt":**
   - 88% clip Voices và 72% clip Music bị cắt băng cứng ở 11,025 kHz, tức nguồn gốc là audio 22,05 kHz. Các lớp ghi hiện trường phần lớn có đủ băng thông đến 16–22 kHz.
   - Mức peak-normalize khác nhau theo lớp: 76% clip Glass breaking và 66% clip Voices có peak ≈ 0 dBFS, trong khi các lớp Crows, Cicadas, Cat và Wind turbine gần như không có clip nào như vậy.
   - Hai lớp có độ dài cố định: 61/109 clip Glass breaking dài đúng 60,5 s, 93/100 clip Wind turbine dài đúng 10,0 s.
6. **Các cụm lớp có phổ chồng lấn**, dựa trên đường bao log-Mel và một probe tuyến tính:
   - động cơ, tiếng ầm tần thấp: Train, Thunder/fireworks/gunshot, Vehicle pass-by, Vehicle idling;
   - máy bay: Propeller aircrafts và Jet aircrafts;
   - tiếng kêu động vật và âm hài ở dải trung: Sirens, Cat, Chicken coop, Crows, Dog;
   - máy móc chạy đều: Vacuum/fan/hairdryer, Lawn mower, Workshop, Bells.

   Các cặp bị probe nhầm nhiều nhất là Propeller→Jet (22%), Vacuum→Workshop (15%), Cat→Crows (12%) và Vehicle pass-by→Thunder (11%).
7. **Gợi ý cho pipeline:**
   - Resample về **22,05 kHz** thay vì 16 kHz. Ở 16 kHz, 23% số clip mất hơn 1% năng lượng; riêng Glass breaking có median 24% năng lượng nằm trên 8 kHz. Ở 22,05 kHz, mép cắt 11,025 kHz của Voices/Music trùng với tần số Nyquist nên mất tác dụng như đường tắt.
   - Không dùng 44,1 kHz trực tiếp.

## 1. Mô tả dataset và quy trình truy cập

| Thuộc tính | Giá trị đo được |
|---|---|
| Số file | 5.048 file `.wav`, không lỗi đọc |
| Định dạng | 44.100 Hz, mono, PCM 16-bit (đồng nhất 100%) |
| Tổng thời lượng | 85.350 s = 23,71 giờ |
| Taxonomy | 22 macro-class (thư mục cấp 1), 28 subclass (thư mục cấp 2 trong 10 macro-class) |
| Tên file | `<Class hoặc Subclass>-NNNN.wav`, ví dụ `Workshop/drill/drill-0007.wav` |
| `sample_id` dùng trong dự án | đường dẫn tương đối kiểu POSIX tính từ `DATASEC/`, ví dụ `Birds/Birds-0001.wav` |

**Nguồn gốc theo bài báo [1].** Phần lớn dữ liệu là bản ghi hiện trường bằng máy đo mức âm Class I, số còn lại lấy từ Freesound và AudioSet (CC0/BY/BY-NC/BY-SA). Voices lấy từ corpus tiếng Ý CLIPS, Music lấy từ MTG-Jamendo (CC BY-NC-SA). Tác giả đã cắt bỏ khoảng lặng và các âm chồng lấn, nên mỗi file chỉ chứa một lớp. Dataset không được commit lên repo (`.gitignore` đã loại `DATASEC/`) vì có thành phần NC/SA.

**Đối chiếu với bài báo.** Cả 39 dòng lớp/subclass trong Table 2 của [1] đều khớp với dữ liệu trong máy về số clip, min/max/mean và tổng thời lượng (sai lệch tổng thời lượng dưới 1 s), trừ Music:

| | Bài báo [1] (Zenodo 15340689) | Bản trong máy |
|---|---|---|
| Số clip Music | 245, mọi clip dài đúng 30,0 s | 1.001 clip, dài 10,0–30,6 s |
| Tổng số clip | 4.292 | 5.048 (= 4.292 − 245 + 1.001) |

**Quy trình truy cập và kiểm tra toàn vẹn**, kiểm ngày 10/10/2026, notebook mục 2:

- Nguồn: tải `DATASEC.zip` từ <https://zenodo.org/records/17033970> rồi giải nén vào `DATASEC/` ở thư mục gốc dự án.
- File zip nặng 6.414.663.316 byte, MD5 `29fa9b8cc84cfa69aa4e5e674e780383`. Bản tải về **khớp MD5** với Zenodo.
- Zip chứa đúng 5.048 file `.wav` (1.001 file Music). Cả 5.048 file khớp CRC32 và kích thước với `DATASEC/`; không có file thừa hoặc thiếu.
- Phần mô tả trên Zenodo vẫn ghi "4.292 file, 18 h 26 m" như bài báo, tức chưa cập nhật theo nội dung archive. Đề môn học chép lại con số này.
- Zenodo cho biết 17033970 không phải phiên bản mới nhất của record. Đề chốt 17033970 nên dự án giữ nguyên phiên bản này.

Ghi chú thêm:

- Theo chính Table 2, tổng thời lượng của [1] là khoảng 17 h 25 m. Phần text của bài báo ghi 18 h 26 m, nên con số này không nhất quán ngay trong bài báo.
- Không có metadata nào cho biết 245 clip Music nào thuộc phiên bản trong bài báo: có 965 clip Music dài đúng 30,0 s. Vì vậy không thể tái tạo chính xác bộ 4.292 clip từ archive hiện tại.

## 2. Phân bố lớp và mức mất cân bằng

![Phân bố lớp](../outputs/eda/figures/fig01_class_distribution.png)

| Chỉ số (`imbalance.csv`) | 22 lớp, theo số clip | 20 lớp (bỏ Voices và Music) | 22 lớp, theo thời lượng |
|---|---|---|---|
| Lớn nhất / nhỏ nhất | 1.900 / 50 = **38,0** | 235 / 50 = 4,7 | 30.008 s / 282 s = 106,6 |
| Tỉ trọng lớp lớn nhất | 37,6% (Voices) | 10,9% (Thunder…) | 35,2% (Music) |
| Normalized entropy | 0,74 | 0,96 | 0,81 |
| Số lớp hiệu dụng exp(H) | 9,8 | 18,0 | 12,1 |

Nhận xét:
- Mô hình chỉ cần luôn đoán "Voices" đã đạt 37,6% accuracy. Accuracy vì thế bị hai lớp lớn chi phối, nên **Macro-F1 là metric chính**, và báo cáo kèm F1 của từng lớp.
- Mất cân bằng theo số clip và theo thời lượng không giống nhau. Music chỉ chiếm 19,8% số clip nhưng 35,2% thời lượng. Còn Horn chiếm 1,1% số clip nhưng chỉ có 0,08 h audio, ít nhất bộ dữ liệu.
- Pipeline cắt mỗi clip về một cửa sổ cố định, nên mỗi epoch mỗi clip chỉ đóng góp một mẫu. Vì vậy **mất cân bằng theo số clip mới là thứ ảnh hưởng tới huấn luyện**. Phần audio dư của clip dài chỉ được tận dụng khi cắt ngẫu nhiên.
- **Mất cân bằng bên trong macro-class.** Ví dụ, Gunshot chiếm 141/235 clip của Thunder/fireworks/gunshot, Brush cutter chiếm 86/127 clip của Lawn mower/brush cutter/olive shaker (`subclass_summary.csv`). Với các macro-class này, kết quả phản ánh chủ yếu subclass lớn nhất.
- Lớp nhỏ nhất là Cat fights and moans, với 50 clip. Nếu test chiếm 15% thì lớp này chỉ còn 7–8 clip test, nên F1 từng lớp sẽ dao động mạnh giữa các seed.

## 3. Thời lượng theo lớp

![Thời lượng](../outputs/eda/figures/fig02_duration_by_class.png)

| Macro-class | n | min (s) | max (s) | mean (s) | median (s) | % < 4 s |
|---|---:|---:|---:|---:|---:|---:|
| Voices | 1900 | 0,2 | 29,9 | 3,2 | 2,2 | 72,2 |
| Music | 1001 | 10,0 | 30,6 | 30,0 | 30,0 | 0,0 |
| Thunder fireworks and gunshot | 235 | 0,4 | 174,6 | 9,7 | 2,0 | 61,7 |
| Workshop | 218 | 0,6 | 233,4 | 25,0 | 20,0 | 9,6 |
| Vehicle pass-by | 217 | 1,1 | 303,1 | 10,8 | 5,3 | 24,9 |
| Lawn mower brush cutter and olive shaker | 127 | 7,2 | 248,9 | 43,0 | 38,4 | 0,0 |
| Vehicle idling | 112 | 2,4 | 481,5 | 36,8 | 23,6 | 1,8 |
| Glass breaking | 109 | 0,2 | 60,5 | 36,6 | 60,5 | 22,9 |
| Propeller aircrafts | 108 | 5,0 | 228,1 | 33,8 | 22,0 | 0,0 |
| Crows seagulls and magpies | 106 | 1,5 | 198,4 | 25,5 | 22,7 | 5,7 |
| Sirens and alarms | 105 | 1,9 | 99,1 | 18,1 | 10,4 | 5,7 |
| Jet aircrafts | 103 | 7,1 | 87,3 | 27,8 | 24,9 | 0,0 |
| Wind turbine | 100 | 10,0 | 59,3 | 11,5 | 10,0 | 0,0 |
| Vacuum cleaner fan and hairdryer | 95 | 2,7 | 301,0 | 38,5 | 23,5 | 1,1 |
| Cicadas and crickets | 74 | 4,2 | 77,4 | 22,9 | 18,2 | 0,0 |
| Dog barkings and howlings | 70 | 0,4 | 37,7 | 9,6 | 7,6 | 32,9 |
| Birds | 69 | 2,0 | 80,0 | 13,2 | 5,0 | 10,1 |
| Bells | 67 | 2,7 | 73,1 | 28,6 | 29,6 | 10,4 |
| Train | 67 | 3,9 | 230,1 | 35,2 | 19,4 | 1,5 |
| Chicken coop | 58 | 1,0 | 55,1 | 12,0 | 11,3 | 32,8 |
| Horn | 57 | 0,3 | 45,2 | 4,9 | 1,9 | 75,4 |
| Cat fights and moans | 50 | 2,4 | 67,0 | 23,4 | 20,0 | 4,0 |
| **Toàn bộ** | **5048** | **0,2** | **481,5** | **16,9** | **8,7** | **34,3** |

Bảng đầy đủ, có thêm std, % < 1 s, % < 5 s và % > 30 s, nằm ở `class_summary.csv`; bảng theo subclass nằm ở `subclass_summary.csv`.

Hệ quả:
- **Chọn 4 s hay 5 s ít ảnh hưởng.** Tỉ lệ clip phải đệm chỉ tăng từ 34,3% lên 38,3%. Cửa sổ 5 s giữ được nhiều nội dung hơn từ các clip dài.
- **Thời lượng tương quan với lớp**, đúng như bản chất vật lý của từng loại âm: Gunshot và Horn ngắn, còn máy móc và máy bay dài. Nếu đệm bằng số 0, tỉ lệ khung im lặng sẽ lộ ra thời lượng gốc, tạo thành một dấu hiệu nhận lớp ngoài nội dung âm. Cần chọn một cách đệm rồi ghi vào config, và đây cũng là biến cần kiểm soát khi trả lời câu hỏi giữa kỳ số 3 về CRNN với các thời lượng khác nhau. Có hai lựa chọn: đệm số 0 rồi phân tích lỗi theo nhóm thời lượng, hoặc đệm bằng cách lặp lại tín hiệu (repeat padding).
- **Clip dài** (Glass breaking 60,5 s, Vehicle idling tới 481 s): khi train nên cắt ngẫu nhiên; khi val/test nên cắt cố định (ví dụ cửa sổ giữa clip, hoặc lấy trung bình nhiều cửa sổ) để kết quả tái lập được.

## 4. Ba biểu diễn tín hiệu: khác biệt toán học và vật lý

![Ví dụ 4 lớp tương phản](../outputs/eda/figures/fig03_examples_waveform_spectrogram_logmel.png)

![Từng bước biến đổi](../outputs/eda/figures/fig04_representation_pipeline.png)

Bốn ví dụ trong hình 3 được chọn theo cấu trúc âm học tương phản; mỗi lớp lấy clip có thời lượng gần 5 s nhất, ổn định qua các lần chạy:
- Gunshot: xung tức thời, dải rộng;
- Sirens: cấu trúc hài, điều tần;
- hairdryer: nhiễu dải rộng dừng (stationary);
- Birds: tiếng hót hài ở tần cao.

**(1) Dạng sóng thô.** $x[n] = x(nT_s)$ với $T_s = 1/f_s$. Đây là áp suất âm đã đổi sang đơn vị số trong [−1, 1]. Vì các file đã qua chỉnh sửa và chuẩn hóa, biên độ chỉ mang ý nghĩa tương đối (dBFS), không còn là dB SPL tuyệt đối. Dạng sóng giữ đầy đủ thông tin, gồm cả pha, nhưng tần số chỉ hiện ra gián tiếp qua dao động. Một hàng gồm 5 s × 22.050 mẫu: mạng phải tự học các bộ lọc, và hai âm giống hệt nhau về mặt nghe có thể có dạng sóng khác hẳn nhau khi lệch pha.

**(2) Spectrogram tần số tuyến tính.**

$$X[k,m] = \sum_{n=0}^{N-1} x[n + mH]\, w[n]\, e^{-j2\pi kn/N}, \qquad S[k,m] = |X[k,m]|^2$$

Ký hiệu và tham số dùng ở đây:
- $w$ là cửa sổ Hann;
- $N = 1024$ mẫu, tương ứng 46,4 ms ở 22,05 kHz;
- bước nhảy $H = 256$ mẫu (11,6 ms);
- bin $k$ ứng với tần số $f_k = k f_s/N$, cách đều $\Delta f = 21{,}5$ Hz.

Bình phương độ lớn bỏ đi pha và cho ra mật độ năng lượng trên mặt phẳng thời gian–tần số. Kích thước $N$ quyết định đánh đổi giữa độ phân giải thời gian và tần số (nguyên lý bất định Gabor): $N$ lớn tách được hài sát nhau nhưng làm nhòe xung Gunshot theo thời gian.

Hình 4(b) cho thấy $S$ hiển thị ở thang tuyến tính gần như trắng hoàn toàn: chỉ vài bin quanh tần cơ bản của còi chiếm gần hết năng lượng, vì dải động vượt quá 80 dB. Đó là lý do mọi hình phổ đều phải hiển thị ở thang dB.

**(3) Log-Mel spectrogram.**

$$M[b,m] = \sum_k H_b[k]\, S[k,m], \qquad L[b,m] = 10\log_{10}\big(M[b,m] + \varepsilon\big)$$

Ở đây $H_b$ là $B = 128$ bộ lọc tam giác đặt cách đều trên thang Mel (hình 4d). Dự án dùng công thức Slaney: tuyến tính dưới 1 kHz, logarit trên 1 kHz; công thức HTK tương ứng là $m = 2595\log_{10}(1 + f/700)$.

Ý nghĩa vật lý và tri giác:
- **Thang Mel** mô phỏng băng tới hạn của ốc tai: độ phân giải tần số mịn ở tần thấp, thô dần ở tần cao. Ở hình 4(e), các hài thấp của còi vẫn tách rõ, còn hài cao bị gộp lại. Số chiều giảm từ 513 bin xuống 128 bin, nên spectrogram nhỏ hơn khoảng 4 lần.
- **Phép log** nén dải động (cảm nhận độ to gần tỉ lệ với log cường độ, định luật Weber–Fechner). Nó cũng biến hiệu ứng nhân thành hiệu ứng cộng: đổi gain $g$ dB làm mọi bin tăng đúng $g$ dB, và đáp ứng của micro trở thành một độ lệch cộng theo tần số.
- **Nhiễu cộng thì không như vậy:** $\log(S + N) \neq \log S + \text{const}$. Nhiễu nền lấp các bin năng lượng thấp và làm mất cấu trúc hài, nên nhiễu SNR được dự đoán sẽ gây hại nhiều hơn nhiễu gain (giả thuyết cho RQ1).
- Hệ quả trực tiếp cho thiết kế: nếu chuẩn hóa từng clip bằng cách trừ trung bình log-Mel, nhiễu gain bị triệt tiêu **hoàn toàn**. Xem mục 7.

## 5. Đường bao phổ và các lớp chồng lấn

![Đường bao phổ theo lớp](../outputs/eda/figures/fig05_class_spectral_envelopes.png)

**Cách tính đường bao phổ.** Với mỗi clip, lấy trung bình theo thời gian của log-Mel (128 band), rồi trừ trung bình theo band để bỏ mức âm. Đường bao của một lớp là trung bình của các clip trong lớp. Khoảng cách giữa hai lớp là RMS của chênh lệch dB trên 128 band (`envelope_distance_db.csv`).

Lúc đầu mình thử dùng tương quan Pearson, nhưng chỉ số này bão hòa: gần như mọi lớp có r > 0,9, vì âm thanh môi trường nói chung đều có phổ dốc xuống ở tần cao.

![Khoảng cách giữa các lớp](../outputs/eda/figures/fig06_class_envelope_distance.png)

![Các cặp giống nhất](../outputs/eda/figures/fig07_most_similar_pairs.png)

Các cụm lớp (phân cụm average-linkage trên khoảng cách đường bao):

| Cụm | Lớp | Đặc điểm phổ | Cặp gần nhất (RMS dB) |
|---|---|---|---|
| Ầm tần thấp, dải rộng | Train, Thunder/fireworks/gunshot, Vehicle pass-by, Vehicle idling (cùng nhánh với Voices, Music) | năng lượng dồn dưới 1 kHz, giảm đều về tần cao | Thunder–Train **1,2**; Thunder–pass-by 2,1; Train–pass-by 2,2; Train–idling 2,4 |
| Máy móc dừng | Vacuum/fan/hairdryer, Bells, Lawn mower…, Horn | phổ tương đối phẳng đến 4 kHz | Bells–Vacuum 1,9; Lawn mower–Vacuum 2,3 |
| Tiếng kêu / âm hài dải trung | Sirens, Cat, Chicken coop, Crows (và Dog) | đỉnh năng lượng 0,5–2 kHz | Cat–Sirens 2,4; Cat–Chicken 2,4; Crows–Sirens 2,5 |
| Tần cao | Glass breaking, Cicadas and crickets, Birds, Workshop | năng lượng mạnh trên 2–4 kHz | Workshop–Birds 3,0 |
| Rất trầm | Propeller, Jet, Wind turbine | dốc rất mạnh, gần như không có năng lượng trên 4 kHz | Propeller–Jet 3,0 |

**Probe khả năng tách lớp** (hình 9, `probe_*.csv`). Để biết lớp nào *thực sự* khó tách, mình dùng một logistic regression cố định trên thống kê log-Mel ở mức clip: mean (đã bỏ mức âm) và std theo thời gian của 128 band. Probe được đánh giá bằng 5-fold stratified CV trên toàn bộ dataset.

> **Lưu ý:** đây chỉ là công cụ chẩn đoán, không phải baseline. Probe không tinh chỉnh tham số nào, không dùng để chọn mô hình, và CV không gom nhóm theo bản ghi gốc nên kết quả có thể lạc quan.

![Probe confusion](../outputs/eda/figures/fig09_probe_confusion.png)

- Kết quả: accuracy 0,808, **Macro-F1 0,671**.
- F1 thấp nhất: Vacuum/fan/hairdryer 0,29, Train 0,43, Propeller 0,43, Vehicle idling 0,47, Sirens 0,48.
- F1 cao nhất: Wind turbine 0,98, Voices 0,98, Music 0,89, Glass breaking 0,81.

Các cặp bị nhầm nhiều nhất (tính theo tỉ lệ của lớp thật):

| Lớp thật → dự đoán | Tỉ lệ | Số clip |
|---|---:|---:|
| Propeller aircrafts → Jet aircrafts | 22,2% | 24 |
| Vacuum cleaner fan and hairdryer → Workshop | 14,7% | 14 |
| Cat fights and moans → Crows seagulls and magpies | 12,0% | 6 |
| Jet aircrafts → Propeller aircrafts | 11,7% | 12 |
| Vehicle pass-by → Thunder fireworks and gunshot | 10,6% | 23 |
| Train → Vehicle pass-by | 10,4% | 7 |
| Dog barkings and howlings → Sirens and alarms | 10,0% | 7 |
| Lawn mower… → Vacuum cleaner… | 8,7% | 11 |
| Sirens and alarms ↔ Chicken coop | 8,6% mỗi chiều | 9 / 5 |

**Nhận xét.** Đường bao phổ trung bình không đủ để dự đoán lỗi. Thunder–Train có đường bao gần nhau nhất (1,2 dB) nhưng probe ít nhầm hai lớp này (Train→Thunder 4,5%, Thunder→Train 2,6%), vì std theo thời gian của chúng khác hẳn: một bên là xung, một bên là âm liên tục. Ngược lại, Propeller–Jet chỉ cách 3,0 dB nhưng bị nhầm nhiều nhất, vì cả đường bao *lẫn* biến thiên theo thời gian (bay ngang chậm) đều giống nhau. Nói cách khác, cấu trúc thời gian mang thông tin phân biệt. Đây là lý do có cơ sở để kỳ vọng CRNN cải thiện nhóm xe, tàu và máy bay so với CNN thuần (câu hỏi giữa kỳ số 3); kỳ vọng này phải được kiểm chứng bằng baseline.

**Dự đoán cho câu hỏi giữa kỳ số 1–2** (giả thuyết, chưa phải kết quả):
- Lớp khó nhất: Vacuum/fan/hairdryer, Train, Propeller, Vehicle idling, Sirens.
- Lớp dễ nhất: Voices, Wind turbine, Music, Glass breaking.
- Các cặp dễ nhầm nghiêm trọng nhất: hai lớp máy bay với nhau; Vacuum với Workshop; nhóm xe với Thunder.

## 6. Chất lượng tín hiệu và dấu vết của khâu tuyển chọn

![Mức âm, rolloff, khung im lặng](../outputs/eda/figures/fig08_signal_level_bandwidth.png)

![Băng thông nguồn](../outputs/eda/figures/fig10_source_bandwidth.png)

| Phát hiện | Số liệu | Rủi ro |
|---|---|---|
| **Băng thông nguồn khác nhau theo lớp** | Mép cắt cứng ở 11,025 kHz: Voices 88,3%, Music 71,5%; ở 16 kHz: Glass breaking 67,9%; phần lớn lớp ghi hiện trường không có mép cắt (`source_bandwidth_by_class.csv`) | Ở 44,1 kHz mô hình có thể nhận Voices/Music chỉ nhờ "không có năng lượng trên 11 kHz". Resample về ≤ 22,05 kHz loại bỏ được dấu hiệu này. |
| **Năng lượng tần cao** | 23,1% clip có trên 1% năng lượng nằm trên 8 kHz; 9,6% có trên 1% năng lượng nằm trên 11,025 kHz. Median năng lượng trên 8 kHz: Glass breaking 24%, Workshop 6%, Lawn mower 1,8% | Resample về 16 kHz làm mất đáng kể thông tin của Glass breaking, Workshop, Cicadas. |
| **Mức peak-normalize theo lớp** | Clip có peak ≥ −0,1 dBFS: Glass breaking 76%, Voices 66%, Music 33%; Crows, Cicadas, Wind turbine, Cat ≤ 1%. Median RMS theo lớp từ −32,2 dBFS (Birds) đến −17,1 dBFS (Music) | Mức âm tuyệt đối tương quan với lớp vì cách tuyển chọn, chứ không phải vì âm học. |
| **Wind turbine rất đồng nhất** | 93/100 clip dài đúng 10,0 s; median peak −15,4 dBFS; 99% năng lượng dưới khoảng 0,2 kHz (centroid 23 Hz); probe recall 0,99 | Nhiều khả năng chỉ từ ít phiên ghi. Điểm test cao có thể phản ánh độ đồng nhất của dữ liệu hơn là khả năng tổng quát hóa (liên quan Extension 4, DataSED). |
| **Glass breaking cắt theo độ dài cố định** | 61/109 clip dài đúng 60,5 s | Có thể là các đoạn liên tiếp của cùng một bản ghi. Không có metadata để kiểm chứng. |
| **Trùng lặp và chồng lấn** | 16 cặp PCM giống hệt (14 Music, Crows-0035/0050, Glass-0097/0098); 5 cặp prefix (Crows-0017⊂0016, Magpies-0004⊂0003, Magpies-0009⊂0008, Seagulls-0008⊂0007, Dog-0042⊂0002); 3 cặp trùng vài giây đầu (Music-0875/0876, Olive shaker-0014/0015, grinder-0031/0032) | Rò rỉ train/test nếu chia ngẫu nhiên. Đã liệt kê 24 nhóm trong `leakage_groups.csv`. |
| Clipping | 31 file có trên 0,1% mẫu chạm full-scale (Music 12 file; Sirens có tỉ lệ cao nhất, 3,8% số clip) | Nhỏ; ghi nhận. |
| DC offset | 72 file có \|DC\| > 0,01 | Nhỏ; bộ lọc thông cao hoặc STFT gần như loại bỏ được. |
| Khung im lặng | Median tỉ lệ khung thấp hơn đỉnh trên 40 dB: Dog 14%, Glass 8%, Voices 4%, các lớp khác ≈ 0 | Đúng với mô tả "đã cắt bỏ khoảng lặng" của [1]; Dog có khoảng nghỉ giữa các tiếng sủa. |

Kiểm tra trùng lặp chỉ phát hiện được nội dung **giống hệt từng mẫu** hoặc **cùng điểm bắt đầu**. Hai đoạn cắt từ cùng một bản ghi gốc nhưng không chung điểm đầu sẽ không bị phát hiện (xem mục 8).

## 7. Hệ quả cho thiết kế thí nghiệm

| Quyết định | Đề xuất | Căn cứ |
|---|---|---|
| Tần số lấy mẫu | **22,05 kHz** (Mel `fmax` = 11.025 Hz) | Giữ được phần lớn năng lượng tần cao (mục 6); mép cắt của Voices/Music trùng Nyquist nên mất tác dụng như đường tắt; vẫn nhẹ cho Colab T4. |
| Độ dài cửa sổ | 5 s (4 s cũng được; chênh lệch nhỏ) | 34% clip dưới 4 s, 38% dưới 5 s; 5 s giữ nhiều nội dung hơn từ clip dài. |
| Đệm / cắt | Train: cắt ngẫu nhiên. Val/test: cắt cố định ở giữa clip. Chọn **một** cách đệm cho mọi mô hình, ghi vào config | Thời lượng tương quan với lớp; đệm số 0 làm lộ thời lượng. |
| Chuẩn hóa | Ưu tiên chuẩn hóa log-Mel bằng mean/std **tính trên tập train** (toàn cục), không chuẩn hóa theo từng clip | Mức âm là dấu vết tuyển chọn. Chuẩn hóa từng clip triệt tiêu luôn nhiễu gain, khi đó không nên chọn gain làm một trong 3 loại nhiễu của RQ1. Nếu vẫn chuẩn hóa từng clip thì ghi rõ lý do và bỏ nhiễu gain. |
| Split | Stratified theo macro-class (tốt hơn: theo subclass), seed cố định, ví dụ 70/15/15. **Gom 24 nhóm trong `leakage_groups.csv`** vào cùng split, hoặc bỏ các bản sao giống hệt | Trùng lặp và prefix gây rò rỉ; Cat chỉ có 50 clip nên test chỉ còn 7–8 clip. |
| Metric | Macro-F1 chính, kèm accuracy, F1 từng lớp và confusion matrix; nên chạy ≥ 3 seed | Voices + Music chiếm 57,5% số clip. |
| Huấn luyện | Class-weighted cross-entropy hoặc balanced sampler; ghi rõ trong config | Tỉ lệ mất cân bằng 38:1. |
| RQ1, giả thuyết cần kiểm chứng | Nhiễu nền/đường phố hại nhiều nhất nhóm ầm tần thấp (xe, tàu, máy bay); che thời gian hại nhóm xung (Gunshot, Glass, Dog); che tần số hại nhóm hài (Sirens, Birds, Bells) | Suy từ đường bao phổ và tỉ lệ khung im lặng ở mục 5–6. |

## 8. Hạn chế của phân tích này

- **Số liệu theo archive thực tế, không theo mô tả.** Mọi số liệu ở đây là của archive Zenodo 17033970 (5.048 clip, đã kiểm MD5/CRC32), không phải con số 4.292 trong đề. Nếu giảng viên muốn dùng đúng 4.292 clip thì không tái tạo được, vì không xác định được 245 clip Music gốc.
- **Không có ID bản ghi gốc**, nên không gom nhóm được các đoạn cắt từ cùng một phiên ghi mà không chung điểm đầu (có thể gặp ở Glass breaking 60,5 s, Wind turbine 10 s, và các file đánh số liên tiếp). Kết quả CV và test với split ngẫu nhiên vì thế có thể lạc quan.
- **Bộ dò mép cắt băng thông là heuristic**: chỉ xét PSD giảm trên 25 dB tại 6 mốc chuẩn. Bộ dò có thể bỏ sót trường hợp lọc dốc thoải, hoặc báo nhầm với âm tự nhiên có phổ tụt đột ngột.
- **Probe dùng toàn bộ dataset** để chẩn đoán khả năng tách lớp, không dùng split. Nó trả lời "lớp nào chồng lấn trên đặc trưng đơn giản", không ước lượng hiệu năng của mô hình thật.
- **Đường bao phổ là trung bình theo thời gian**, nên bỏ qua các mẫu hình theo thời gian (điều tần của còi, nhịp sủa). Probe chỉ bù một phần thông qua std theo thời gian.

## 9. Tái lập

Cách chính là mở `notebooks/01_eda_datasec.ipynb` và chạy *Run All*. Lần đầu mất khoảng 2 phút trên CPU 16 luồng; các lần sau dùng lại cache. Có thể chạy không giao diện hoặc dùng script dòng lệnh:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_datasec.ipynb
python scripts/eda_datasec.py --data DATASEC --out outputs/eda --jobs 10   # cùng phân tích, kết quả giống hệt
python scripts/eda_datasec.py --plots-only                                # chỉ vẽ lại từ cache
```

- **Tham số trích đặc trưng:** 44,1 → 22,05 kHz bằng `scipy.signal.resample_poly(1, 2)`; STFT Hann với `n_fft = 1024`, hop 256, center + reflect padding; 128 Mel band (Slaney, chuẩn hóa diện tích), 0–11.025 Hz; log $10\log_{10}(\max(P, 10^{-10}))$; PSD Welch ở 44,1 kHz với `nperseg = 4096`; seed 42. Danh sách đầy đủ và phiên bản thư viện nằm trong `outputs/eda/eda_run.json`.
- **Môi trường đã chạy:** Python 3.13.9, numpy 2.3.4, scipy 1.16.3, pandas 2.3.3, soundfile 0.14.0, matplotlib 3.10.8, scikit-learn 1.7.2; Windows 11, CPU 16 luồng; không dùng GPU.
- **Kết quả xuất ra:**
  - `outputs/eda/tables/file_features.csv`: một dòng mỗi clip, gồm `sample_id`, lớp, subclass, thời lượng, md5 và các chỉ số tín hiệu. Có thể dùng làm manifest khi tạo split.
  - `outputs/eda/tables/leakage_groups.csv`: các nhóm clip phải nằm cùng một split.
  - `outputs/eda/cache/`: ma trận mean/std log-Mel; không commit, sinh lại được bằng script.

## Tài liệu tham khảo

[1] L. Fredianelli et al., "Environmental Noise Dataset for Sound Event Classification and Detection," *Scientific Data*, vol. 12, 1712, 2025. doi:10.1038/s41597-025-05991-w
