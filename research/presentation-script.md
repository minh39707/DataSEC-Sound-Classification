# Kịch bản thuyết trình midterm (tiếng Việt)

Slide viết bằng tiếng Anh, lời nói bằng tiếng Việt. Sinh tự động bởi `scripts/build_presentation.py`; sửa nội dung trong file đó rồi chạy lại. Mục có dấu [CẦN BẠN LÀM] là chỗ tôi không thể làm thay.

## Slide 1: DataSEC: Environmental Sound Classification under Noise

Chào thầy/cô và các bạn. Nhóm em làm track 3: phân loại âm thanh môi trường 22 lớp trên bộ DataSEC. Hôm nay em trình bày phần giữa kỳ: hiểu dữ liệu, pipeline, ba mô hình nền, và phân tích lỗi; cuối cùng là câu hỏi nghiên cứu cho phần Final.

## Slide 2: Dataset: 5,048 clips, 22 classes, heavily imbalanced

Bộ dữ liệu có 5.048 file, nhiều hơn con số 4.292 trong đề vì bản Zenodo ghim DOI là bản mới; nhóm đã ghi rõ chênh lệch này. Mất cân bằng rất nặng: lớp lớn nhất gấp 38 lần lớp nhỏ nhất. Chỉ cần đoán toàn bộ là Voices đã được accuracy 0,376 nhưng macro-F1 chỉ 0,025. Vì vậy chúng em dùng macro-F1 làm thước đo chính và dùng loss có trọng số theo lớp.

## Slide 3: Durations vary from 0.2 s to 8 min, and duration leaks the class

Thời lượng rất khác nhau giữa các lớp, từ 0,2 giây đến 8 phút. Chỉ dùng thời lượng để đoán đã cho macro-F1 0,272, cao hơn ngẫu nhiên nhiều, nghĩa là mô hình có thể học độ dài phần đệm thay vì học âm thanh. Ngoài ra Voices và Music bị cắt cứng ở 11 kHz, nên ta resample về 22,05 kHz để xoá dấu vết này. Đây là các rủi ro shortcut mà chúng em nhớ khi đọc kết quả.

## Slide 4: Waveform vs linear spectrogram vs log-Mel

Waveform cho biết khi nào to nhỏ nhưng không đọc được tần số. Spectrogram tuyến tính có 513 bin chia đều, phần lớn là tần số cao ít thông tin. log-Mel gom thành 64 dải theo thang Mel, dải thấp hẹp dải cao rộng giống tai người, rồi lấy log để nén dải động. Một hệ quả quan trọng: đổi âm lượng chỉ là cộng một hằng số trong miền log, điều này liên quan đến thí nghiệm gain ở phần Final.

## Slide 5: Which classes overlap spectrally?

Chúng em tính độ giống nhau của phổ trung bình giữa các lớp. Cặp giống nhất là Jet và Propeller, rồi Train. Chúng hình thành nhóm động cơ/xe cộ. Tuy nhiên giống phổ chưa chắc bị nhầm: Glass breaking và Workshop giống 0,98 nhưng không bao giờ bị nhầm, vì tiếng thủy tinh là xung ngắn còn phổ trung bình đã mất thông tin thời gian.

## Slide 6: Pipeline and split

Pipeline chạy lúc chạy, không lưu ảnh spectrogram: chỉ cache âm thanh int16. Split chia theo 40 lớp con nên lớp nhỏ nhất vẫn có 8 clip test. Chọn checkpoint bằng macro-F1 trên tập validation, không bao giờ chỉnh theo test. Em cũng nói thẳng: có 2 clip test trùng một phần âm thanh với train; loại chúng ra thì macro-F1 đổi không quá 0,0017 nên giữ nguyên split.

## Slide 7: Three baseline models

Ba mô hình theo đề: CNN nhỏ, MobileNetV3-Small pretrained trên ImageNet, và CRNN có GRU hai chiều. Cùng một công thức huấn luyện: AdamW, cosine, loss trọng số theo lớp, dừng sớm theo macro-F1 validation. Chạy ba seed.

## Slide 8: Baseline results: three seeds, no clear winner

Đây là kết quả chính. Với ba seed, ba mô hình gần như ngang nhau: 0,876, 0,901 và 0,875, độ lệch giữa seed lên tới 0,027 bằng khoảng cách giữa các mô hình. Em muốn nói thẳng: lần chạy đầu trên Colab chỉ một seed cho thứ hạng khác, CRNN thấp nhất 0,815, nhưng lần đó CRNN dừng sớm ở epoch 20. Khi chạy lại ba seed thì thứ hạng đó không lặp lại, nên nhóm không kết luận mô hình nào tốt hơn trên dữ liệu sạch. Latency đo trên RTX 2060 SUPER, không so được với số T4 trước đó.

## Slide 9: Per-class F1: best and worst classes

Câu hỏi E1. Tốt nhất là Voices, Chicken coop, Wind turbine, Music, Cicadas với F1 từ 0,97. Tệ nhất là Jet, Train, Vehicle idling, Propeller và Vacuum. Nhóm khó khá ổn định giữa các seed. Nhưng cần thận trọng: ba lớp điểm cao có dấu vết xử lý dữ liệu như cắt ở 11 kHz hay độ dài cố định, nên điểm cao chưa chắc là đã học đúng âm thanh. Biểu đồ trên là seed 42.

## Slide 10: Error analysis: engine-like sounds confuse each other

Câu hỏi E2. Các cặp nhầm nhiều nhất đều là tiếng máy móc, động cơ liên tục: Jet với Propeller, Lawn mower với Vehicle idling, Vacuum với Workshop. Ngoài nhóm này, lỗi đáng kể nhất là Music bị đoán thành Bells. Ma trận nhầm lẫn bên phải là small_cnn seed 42. [CẦN BẠN LÀM] Phần nghe thử các clip bị nhầm chưa được điền; trước khi trình bày hãy nghe vài clip trong bảng E2 để nêu nguyên nhân thật.

## Slide 11: Does the GRU help? CRNN − CNN by clip duration

Câu hỏi E3. Chênh lệch CRNN trừ CNN theo độ dài clip nhỏ và đổi dấu giữa các seed, nên không có cải thiện đo được từ GRU. Kết luận từ lần chạy một seed là CRNN kém hơn ở clip dài không còn đúng. Hơn nữa CRNN và CNN khác cả phần conv nên đây không phải phép thử sạch cho tác dụng của GRU.

## Slide 12: Performance–cost trade-off

Câu hỏi E4. Không có người thắng tuyệt đối, tuỳ tiêu chí chi phí. MobileNet có F1 cao nhất và ít MACs nhất nhưng chậm nhất trên GPU ở batch 1. small_cnn nhanh nhất và nhỏ nhất. CRNN bị trội trên đường Pareto theo độ trễ và số tham số ở dữ liệu sạch. Chúng em chưa đo VRAM đỉnh.

## Slide 13: Proposed Final question (RQ1) and design

Câu hỏi cho phần Final là câu RQ1 của đề: ba mô hình mất bao nhiêu khi tập test bị làm bẩn. Giữ nguyên split, checkpoint và pipeline, không huấn luyện lại, chỉ thay đổi đầu vào lúc test. Bốn kiểu làm bẩn, mỗi kiểu ba mức. Kết quả sẽ báo cáo F1 sạch, F1 bẩn, ΔF1 kèm đường suy giảm và phân tích theo lớp.

## Slide 14: Limitations

Hạn chế: số Colab một seed và số chạy local ba seed không trộn lẫn; nhiều lớp chỉ có 8 đến 17 clip test; không có mã phiên ghi âm nên có thể còn rò rỉ nhẹ; phần nghe thử lỗi chưa xong. Em nêu thẳng các điểm này thay vì né.
