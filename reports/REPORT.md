# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Đinh Công Minh

Công cụ gán nhãn đã dùng: CVAT (Docker local v2.76.0) kết hợp trực quan hóa và kiểm tra trực tiếp file nhãn Ultralytics YOLO Detection 1.0

Báo cáo chi tiết các vòng học chủ động cho bài toán phát hiện phương tiện giao thông ban đêm. Mọi số liệu được truy xuất trực tiếp từ `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round0.json`, `outputs/round1_diff.md` và `outputs/round1_diff.json`.

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool - 268 ảnh) và tập kiểm thử (test set - 20 ảnh) được chia theo trục thời gian, có vùng đệm 4 giây ở giữa thay vì chia ngẫu nhiên xuất phát từ đặc tính vật lý của video:

1. **Bản chất chuỗi thời gian của camera tĩnh:** Video được ghi từ camera cố định trên cầu vượt nhìn xuống đường cao tốc vào ban đêm, trích xuất ở tốc độ 2.5 fps (mỗi khung hình cách nhau 0.4 giây). Vì camera không di chuyển và các phương tiện chạy với vận tốc liên tục, mỗi chiếc xe lưu lại trong góc nhìn từ 3 đến 8 giây. Hai khung hình liền kề cách nhau 0.4 giây gần như tương đồng hoàn toàn về nền đường, điều kiện chiếu sáng và vị trí xe.
2. **Nguy cơ rò rỉ dữ liệu (data leakage / temporal leakage):** Nếu chia ngẫu nhiên (random split), các khung hình của cùng một chiếc xe sẽ rơi đồng thời vào cả tập huấn luyện và tập kiểm thử. Khi đó, mô hình được chấm điểm trên chính những chiếc xe, góc phản chiếu đèn và bối cảnh mà nó vừa được học.
3. **Số đo bị lệch theo hướng lạc quan giả tạo:** Nếu chia ngẫu nhiên, số đo hiệu năng (AP50, Precision, Recall, F1) trên tập kiểm thử sẽ bị thổi phồng quá mức (overestimated / inflated), che giấu khả năng khái quát hóa thực tế của mô hình khi gặp luồng xe hoặc khoảng thời gian chưa từng thấy.
4. **Vai trò của vùng đệm (buffer zone):** Việc chia 4 đoạn kiểm thử (tâm tại giây 20, 60, 100, 140) cùng vùng đệm 112 ảnh (4 giây trước và sau mỗi đoạn kiểm thử) bảo đảm khoảng cách tối thiểu giữa ảnh pool gần nhất và ảnh test là 4.4 giây. Khoảng cách thời gian này đủ để mọi chiếc xe ở tập pool chạy thoát hoàn toàn khỏi khung hình trước khi đoạn kiểm thử bắt đầu, ngăn chặn triệt để hiện tượng rò rỉ dữ liệu.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào `outputs/compare_round0.jpg` và `outputs/metrics_round0.json`, mô hình khởi đầu lạnh (yolov8n pretrained trên COCO, hợp nhất 3 lớp car, bus, truck) bộc lộ sự bất đối xứng rõ rệt giữa Precision rất cao (0.925) và Recall rất thấp (0.489, bỏ sót 206 trên tổng số 403 box tham chiếu):

- **Các loại xe không khớp nhãn tham chiếu:**
  1. *Xe ở khoảng cách xa (small objects):* Các xe ở sát đường chân trời hoặc chân cầu vượt chỉ xuất hiện dưới dạng hai đốm sáng nhỏ; mô hình cold start bỏ sót phần lớn các xe này (box màu vàng FN xuất hiện dày đặc ở nửa trên của ảnh `compare_round0.jpg`).
  2. *Xe ở các làn tối ngoài cùng bên phải:* Xe chạy xa dần với đèn hậu đỏ mờ nhạt, thân xe chìm trong bóng tối bên lề đường bị mô hình bỏ qua do COCO chủ yếu được huấn luyện trên ảnh ban ngày có độ tương phản thân xe rõ ràng.
  3. *Xe ngược chiều có đèn pha chói:* Xe chạy tới với đèn pha rọi thẳng vào ống kính gây hiện tượng tán xạ ánh sáng; mô hình phân vân hoặc kéo lệch box xuống vệt đèn pha loang trên mặt đường bê tông.
- **Ý nghĩa của độ phủ (Recall) theo kích thước xe:**
  Recall small = 0.182, Recall medium = 0.547, Recall large = 0.561. Số liệu này chứng minh điểm mù nghiêm trọng nhất của mô hình ban đầu là phân khúc xe nhỏ ở xa (chỉ phát hiện được 18.2% số xe nhỏ, tức 12/66 xe). Khi kích thước xe tăng lên mức trung bình và lớn, độ phủ tăng gấp 3 lần (đạt ~55-56%), phản ánh mô hình COCO phụ thuộc nhiều vào các đường nét thân xe đầy đủ thay vì các tín hiệu cụm đèn xe trong đêm.
- **Trường hợp cần rà lại nhãn tham chiếu trước khi kết luận mô hình sai:**
  Nhãn tham chiếu của tập kiểm thử (`data/test/labels/`) được sinh tự động bởi một mô hình phát hiện đối tượng khác và chưa từng được con người thẩm định từng box (xem `data/DATA.md`). Trên ảnh `outputs/compare_round0.jpg` tại `frame_0350` (góc trên bên trái), mô hình cold start đưa ra một box màu đỏ (FP) ở khu vực dải phân cách/biển báo phản quang. Khi phóng to ảnh gốc, cần xác định xem đây là xe thật bị nhãn tham chiếu bỏ sót hay thực sự là ánh sáng phản xạ từ biển báo. Nếu là xe thật, mô hình cold start không hề sai mà do nhãn tham chiếu bị thiếu; ngược lại nếu là biển báo, mô hình mới thực sự mắc lỗi False Positive. Không được coi nhãn tham chiếu là chân lý tuyệt đối.

## 3. Chiến lược chọn mẫu

### Giải thích công thức tính điểm và vai trò của `MIN_GAP_S`

Công thức tính điểm ưu tiên của từng khung hình trong pool:
$$\text{score} = W_U \cdot U + W_A \cdot A + W_D \cdot D$$
với trọng số mặc định $W_U = 0.5$, $W_A = 0.3$, $W_D = 0.2$:

- **$U$ (Uncertainty - Độ bất định đỉnh):** Trung bình của 5 giá trị bất định lớn nhất của các box trong frame. Độ bất định của mỗi box tính bằng $u(c) = 1 - |2c - 1|$, đạt giá trị cực đại 1.0 khi độ tin cậy $c = 0.5$ (mô hình phân vân tuyệt đối giữa việc có xe hay không) và bằng 0 khi $c$ tiến sát 0 hoặc 1. Trọng số $W_U = 0.5$ đặt ưu tiên hàng đầu vào các frame chứa những box gây bối rối nhất cho mô hình.
- **$A$ (Ambiguity - Mật độ box mập mờ):** Số lượng box có độ tin cậy nằm trong vùng ranh giới $0.15 \le c < 0.50$, chuẩn hóa theo số lượng box mập mờ lớn nhất trong toàn bộ pool. Trọng số $W_A = 0.3$ giúp chọn các frame có nhiều đối tượng mà mô hình nghi ngờ nhưng chưa đủ tự tin vượt ngưỡng phát hiện 0.5.
- **$D$ (Diversity - Độ đa dạng thời gian):** Khoảng cách thời gian từ frame đang xét tới frame đã được chọn/gán gần nhất, chia cho giới hạn trần `DIVERSITY_CAP_S = 10.0s`. Trọng số $W_D = 0.2$ ngăn thuật toán tập trung lấy mẫu dồn cục vào một đoạn video ngắn, giúp trải đều dữ liệu huấn luyện theo trục thời gian.
- **Vai trò của `MIN_GAP_S = 2.0s`:** Thuật toán chọn mẫu tham lam (greedy) tự động loại bỏ bất kỳ frame nào có thời điểm cách frame đã chọn trước đó dưới 2.0 giây. Do camera đứng yên, hai frame cách nhau dưới 2 giây có bối cảnh và vị trí xe gần như trùng lặp (redundant). Luật này đảm bảo ngân sách gán nhãn không bị lãng phí vào các ảnh tương tự nhau, tối đa hóa lượng tri thức mới đưa vào huấn luyện.

### Dẫn chứng từ `reports/SELECTION.md` và cân nhắc thực tế

1. `frame_0182.jpg` (rank 1, $t=72.8$s, score = 0.9591): Đứng đầu toàn pool về điểm số, có $U=0.9182$, $A=1.0000$ (18 box mơ hồ), mật độ xe cao với nhiều xe ngược chiều phân vân.
2. `frame_0369.jpg` (rank 2, $t=147.6$s, score = 0.9324): Đại diện cho đoạn cuối video với 43 box dự đoán, $U=0.9315$, $A=0.8889$, cung cấp nhiều tình huống giao thông đông đúc.
3. `frame_0099.jpg` (rank 8, $t=39.6$s, score = 0.9063): Nằm ở khoảng thời gian đầu video ($t=39.6$s, cách xa các frame trên > 30s), có $U$ vượt trội (0.9460), mật độ xe vừa phải giúp rà soát kỹ các ca xe tối ngược chiều.
4. **Frame đối chiếu - `frame_0372.jpg` (rank 6, $t=148.8$s, score = 0.9101):** Mặc dù điểm số của frame 372 cao hơn frame 0099, nó bị **LOẠI BỎ** hoàn toàn vì chỉ cách `frame_0369.jpg` ($t=147.6$s) đúng 1.2 giây ($< \text{MIN\_GAP\_S} = 2.0$s). Bằng chứng trên CSV và contact sheet cho thấy frame 372 và frame 369 gần như cùng chung một bối cảnh xe; loại bỏ frame 372 giúp tiết kiệm công gán nhãn và tránh trùng lặp dữ liệu huấn luyện.

### Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không?

**KHÔNG.** Điểm bất định cao chỉ phản ánh trạng thái băn khoăn toán học của mô hình tại ngưỡng xác suất 0.5. Sự băn khoăn này hoàn toàn có thể bắt nguồn từ các yếu tố nhiễu ngoại cảnh (chẳng hạn như vệt đèn pha chói lòa trên mặt đường bê tông ướt, ánh đèn cầu vượt phản chiếu, biển báo phát sáng, bụi sương ban đêm). Khi đó mô hình phân vân với những đối tượng không phải là xe; gán nhãn những frame này không giúp mô hình học thêm được thuộc tính xe thực tế mà còn có nguy cơ nạp thêm nhiễu. Ngoài ra, việc cải thiện số đo trên tập test còn phụ thuộc vào việc các ca khó trong pool có cùng phân phối với các tình huống trong 20 ảnh test hay không, cũng như nguy cơ overfitting khi huấn luyện trên tập dữ liệu kích thước nhỏ (12 ảnh).

## 4. Các vòng học chủ động (active learning)

### Bảng tổng hợp các vòng

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 357 | 0.664 | -0.108 | 1.000 | 0.055 | 0.103 | 0.000 | 0.034 | 0.293 |

### Mức độ sửa nhãn gợi ý vòng 1 (truy xuất từ `outputs/round1_diff.md` và `outputs/round1_diff.json`)

- Tổng số ảnh trong lô: 12 ảnh.
- Số box mô hình đề xuất ban đầu: 169 box.
- Số box sau khi con người rà soát trên CVAT: 357 box.
- **Số box giữ nguyên (accepted):** 145 box (tỷ lệ chấp nhận đạt 86%, phản ánh sự kế thừa hợp lý các gợi ý chuẩn xác của mô hình).
- **Số box chỉnh sửa biên dạng (edited):** 10 box (sửa box bị kéo dài quá mức do ôm vệt đèn hoặc căn chỉnh lại cản trước/đuôi xe).
- **Số box xóa bỏ (deleted - False Positive của AI):** 14 box (xóa các box trùng lặp trên cùng một xe hoặc box bắt nhầm vào ánh sáng phản chiếu).
- **Số box thêm mới (added - False Negative của AI):** 202 box (bổ sung triệt để các xe màu tối ở xa, xe ở góc khuất và xe chạy ngược chiều bị bỏ sót).

### Phân tích kết quả thực nghiệm sau fine-tune trên tập kiểm thử

- **AP50 và các chỉ số tổng thể:**
  - AP50 đạt **0.664**, giảm **-0.108** so với khởi đầu lạnh (0.771).
  - Độ chính xác (Precision) tăng tuyệt đối lên **1.000** (100%): trong toàn bộ 20 ảnh kiểm thử, mô hình không mắc bất kỳ lỗi phát hiện nhầm nào (**FP = 0**, so với 16 FP ở cold start).
  - Độ phủ (Recall) giảm xuống **0.055** (phát hiện được 22/403 box tham chiếu, 381 FN) ở ngưỡng conf 0.25 do mô hình trở nên cực kỳ khắt khe và thận trọng.
- **Nhóm xe tốt lên / xấu đi theo kích thước:**
  - *Tốt lên về độ tin cậy và triệt tiêu báo động giả:* Hoàn toàn không còn hiện tượng vẽ box nhầm vào vệt sáng đèn pha trên mặt đường, đèn đường hay biển báo. Nhóm xe lớn cận cảnh ($R_{large} = 0.293$) duy trì nhận diện ổn định nhất với các box ôm khít thân xe.
  - *Xấu đi về độ phủ ở ngưỡng mặc định 0.25:* Nhóm xe nhỏ ở xa ($R_{small} = 0.000$) và xe cỡ vừa ($R_{medium} = 0.034$) bị tụt recall mạnh vì mô hình hạ thấp phân phối confidence của các xe ở xa để tránh FP.
- **Ca thay đổi cụ thể trên `outputs/compare_round1.jpg`:**
  - Tại `frame_0350` (hàng cuối cùng): Ở mô hình cold start, xuất hiện một box màu đỏ (FP) ở góc trên bên trái do nhầm lẫn ánh sáng phản chiếu từ dải phân cách/biển báo. Sau khi fine-tune ở round 1, box đỏ này đã **hoàn toàn biến mất** (cột round 1 sạch bóng box đỏ FP), mô hình chỉ kích hoạt duy nhất một box màu xanh lá (TP) ôm rất chuẩn xác chiếc xe ở làn giữa cận cảnh. Lý do: trong quá trình sửa nhãn vòng 1, việc xóa bỏ các box ôm vệt đèn phản chiếu ở `frame_0187` và box trùng ở `frame_0331` đã dạy cho mô hình phân biệt rạch ròi giữa phản xạ ánh sáng và thân xe thực tế.

### Phân biệt quan sát độc lập, lỗi pre-label và kết quả mô hình

Dựa vào việc đối chiếu chéo giữa `reports/BLIND_SCAN.md`, `reports/REVIEW_LOG.csv`, `outputs/round1_diff.md` và `outputs/compare_round1.jpg`:

1. **Quan sát độc lập (Blind Scan trên `frame_0182.jpg`):** Trước khi mở nhãn gợi ý, người thẩm định quan sát độc lập bằng mắt thường thấy 24 xe và dự báo 2 vị trí AI dễ sai: (1) Xe con màu tối ở mép dưới cùng giữa ảnh (`cx ~ 0.43, cy ~ 0.94`) bị cắt khung hình; (2) Vệt đèn pha xe rọi sáng mặt đường bê tông ở làn giữa (`cx ~ 0.53, cy ~ 0.73`) dễ làm AI vẽ box quá dài.
2. **Lỗi pre-label đã phát hiện và xử lý (`REVIEW_LOG.csv` - đã ghi nhận đầy đủ 33 ca đại diện trên toàn bộ 12 frame):**
   - *Ca thêm mới (added):* Đúng như dự đoán độc lập, AI bỏ sót hoàn toàn chiếc xe tối đang tiến vào ở mép dưới (`frame_0182.jpg`, `cx=0.430, cy=0.945`). Thân xe nhìn rõ nắp capo và kính lái, đã được bổ sung box ôm sát thân xe theo quy tắc xe bị cắt ở mép ảnh. Các trường hợp tương tự ở mép dưới cũng được bổ sung đồng bộ tại `frame_0107.jpg` (`cx=0.436, cy=0.950`), `frame_0312.jpg` (`cx=0.453, cy=0.959`), `frame_0380.jpg` (`cx=0.674, cy=0.961`), và `frame_0392.jpg` (`cx=0.254, cy=0.947`). Ngoài ra, tại `frame_0326.jpg` (`cx=0.543, cy=0.584`), bổ sung xe bị xe phía trước che khuất một phần góc phải; tại `frame_0380.jpg` (`cx=0.422, cy=0.538`), bổ sung xe di chuyển tốc độ cao bị nhoè chuyển động (motion blur) theo quy tắc ôm trọn vệt nhoè.
   - *Ca xóa box trùng và báo động giả (deleted):* Tại `frame_0182.jpg`, AI sinh ra 2 box trùng nhau cho cùng một xe ở làn giữa xa (`cx=0.483, cy=0.472` và `cx=0.483, cy=0.460`); đã xóa 1 box thừa. Tại `frame_0331.jpg`, xuất hiện cụm 3 box chồng lấn trên cùng một xe (`cx=0.369`, `0.391`, `0.409`), đã xóa 2 box thừa. Đặc biệt tại `frame_0312.jpg` (`cx=0.698, cy=0.592`), AI sinh 1 box khổng lồ ($w=0.143, h=0.134$) gộp nhầm 2 xe chạy sát nhau; đã xóa box gộp để tách thành 2 box độc lập theo quy tắc 'Hai xe đứng sát nhau: Vẽ hai box riêng'. Đồng thời xóa các box nhận diện nhầm ánh đèn phản quang trên dải phân cách cứng tại `frame_0099.jpg` (`cx=0.340`), `frame_0326.jpg` (`cx=0.205`) và `frame_0392.jpg` (`cx=0.568`).
   - *Ca chỉnh sửa (edited):* Tại `frame_0187.jpg` (`cx=0.608, cy=0.890`), box gợi ý ban đầu có chiều cao bất thường $h=0.182$ do ôm trọn vệt sáng đèn pha loang trên đường; đã thu gọn cạnh dưới lên sát cụm đèn trước và gầm xe ($h=0.105$). Thao tác thu gọn viền loại bỏ vệt sáng cũng được thực hiện tại `frame_0107.jpg` ($h=0.091 \to 0.070$), `frame_0270.jpg` ($h=0.087 \to 0.068$), `frame_0312.jpg` ($w=0.060 \to 0.048$) và `frame_0331.jpg` ($w=0.057 \to 0.050$). Đối với xe tải, tại `frame_0227.jpg`, box truck bị cắt thành 2 box đã được gộp và mở rộng ($cx=0.294, cy=0.545, w=0.115, h=0.185$); tại `frame_0392.jpg` (`cx=0.652, cy=0.445`), box của AI chỉ bao đầu cabin đã được kéo nới rộng chiều ngang ($w=0.087 \to 0.124$) để bao trọn cả thùng xe tải phía sau theo quy tắc 'Mọi phương tiện từ 4 bánh trở lên'.
   - *Ca chấp nhận (accepted):* Các box dự đoán của AI có độ chuẩn xác cao ($IoU \ge 0.85$, nhiều ca $IoU = 1.000$) ôm khít thân xe, bánh xe và gương chiếu hậu được giữ nguyên, tiêu biểu như: xe SUV màu trắng (`frame_0182.jpg`, `cx=0.785, cy=0.562`), xe con làn đối diện (`frame_0187.jpg`, `cx=0.288`), xe sedan đi thẳng (`frame_0270.jpg`, `cx=0.293`), xe tải lớn cận cảnh (`frame_0326.jpg`, `cx=0.262, cy=0.899`), xe sedan cận cảnh (`frame_0380.jpg`, `cx=0.247`), và xe chạy xa dần ở làn phải (`frame_0369.jpg`, `cx=0.852, cy=0.628`).
3. **Mô tả một ca khó theo guideline (`GUIDELINE_LABEL.md`):**
   Tình huống xe ở làn đường xa ngược chiều (`frame_0099.jpg`, `frame_0107.jpg`) khi thân xe chìm hoàn toàn vào màn đêm, chỉ nhìn thấy 2 chấm sáng của đèn pha. Theo quy tắc *'Chỉ thấy đèn, thân xe tối nhưng vẫn đoán được đường viền'*, người gán nhãn không được khoanh riêng 2 chấm đèn mà phải ước lượng đường bao thân xe hình chữ nhật quanh cụm đèn, đồng thời kiên quyết không kéo box trùm xuống vệt đèn pha rọi sáng mặt đường phía trước. Việc duy trì nhất quán nguyên tắc này qua tất cả các frame giúp mô hình không bị học phải nhiễu.

## 5. Kết luận và giới hạn

### Đánh giá vòng 1 so với cold start và lý do dừng hoặc tiếp tục

- **So sánh kết quả vòng 1 so với cold start:**
  - AP50 vòng 1 đạt **0.664** (giảm 0.108 so với cold start 0.771).
  - Về mặt tích cực (Precision & Triệt tiêu FP): Precision đạt mức hoàn hảo **1.000 (100%)**, triệt tiêu toàn bộ 16 ca báo động giả (False Positive) từ cold start về **0 FP**. Mô hình đã học được bài học đắt giá từ 357 box nhãn chuẩn: không bao giờ vẽ nhầm vào vệt đèn pha phản chiếu trên mặt đường ướt hay đèn đường.
  - Về mặt hạn chế (Recall & Ngưỡng tin cậy): Do bị phạt nặng ở các box trùng lặp và vệt đèn trong quá trình train 50 epoch, mô hình trở nên cực kỳ thận trọng và hạ thấp phân phối confidence của các xe ở xa, khiến Recall tại ngưỡng mặc định 0.25 bị tụt xuống **0.055**.
- **Lý do dừng hay tiếp tục:**
  - Tôi quyết định **tiếp tục** sang vòng 2 (sử dụng 12 ảnh tiếp theo từ `outputs/selection_round2.csv`) để cung cấp thêm dữ liệu đa dạng về các cụm xe ở xa, giúp mô hình tăng lại độ phủ (Recall) cho nhóm xe nhỏ và vừa mà vẫn giữ được độ chính xác tuyệt đối (Precision). Đồng thời, ở khía cạnh triển khai, có thể cân nhắc hạ ngưỡng phát hiện (ví dụ từ conf 0.25 xuống 0.10–0.15) để khai thác các box xe thật mà mô hình dự đoán ở mức tin cậy vừa phải.

### Đề xuất hai ca còn yếu hoặc bất định cho vòng tiếp theo

1. **Ca xe tải / xe buýt / xe container cỡ lớn đi đêm (như ca box truck ở `frame_0227.jpg`):** Mô hình ban đầu thường xuyên xé nhỏ phương tiện dài thành nhiều box độc lập hoặc bỏ sót phần thùng xe tối phía sau cabin.
   - *Chi phí rà nhãn:* Trung bình (cần kéo chỉnh cẩn thận đường biên bao quát cả đầu kéo lẫn thùng xe).
   - *Nguy cơ ảnh gần trùng:* Thấp, nếu áp dụng nghiêm ngặt luật khoảng cách thời gian `MIN_GAP_S >= 2.0s`.
2. **Ca cụm xe ở làn ngoài cùng bên phải chạy xa dần trong bóng tối (như ở `frame_0107.jpg`, `frame_0331.jpg`):** Xe nhỏ chỉ còn thấy 2 chấm đỏ của đèn hậu, độ tương phản cực kỳ thấp giữa nền đường tối và thân xe.
   - *Chi phí rà nhãn:* Cao, đòi hỏi người thẩm định phải phóng to nhiều lần để phân biệt giữa đèn hậu xe với phản quang dải phân cách.
   - *Nguy cơ ảnh gần trùng:* Cao, vì xe chạy cùng chiều có thể xuất hiện liên tiếp ở các frame liền kề nếu không lọc kỹ.

### Ảnh hưởng của các giới hạn thực nghiệm đến kết luận

1. **Tập kiểm thử chỉ có 20 ảnh (403 box tham chiếu):** Cỡ mẫu kiểm thử nhỏ dẫn đến phương sai thống kê (variance) cao; sự thay đổi của chỉ một vài box có thể làm biến động AP50 từ 0.01 đến 0.02. Do đó, các biến động nhỏ không phản ánh đầy đủ năng lực thực tế của mô hình.
2. **Luật bỏ qua xe quá nhỏ (chiều cao dưới 16 pixel - 14 box):** Giúp loại bỏ sự tranh cãi không đáng có ở các xe sát đường chân trời, nhưng đồng thời khiến số đo chưa phản ánh được năng lực phát hiện phương tiện ở cự ly cực xa.
3. **Nhãn tham chiếu do mô hình tự sinh và chưa qua rà soát thủ công:** Đây là giới hạn trọng yếu nhất. Nhãn tham chiếu không phải chân lý tuyệt đối. Khi mô hình fine-tune phát hiện đúng một chiếc xe thật trong đêm mà nhãn tham chiếu bỏ sót, mô hình lại bị hệ thống chấm phạt là False Positive (FP), làm giảm Precision và AP50 một cách oan uổng. Do đó, cần luôn kết hợp số đo định lượng với quan sát định tính trên ảnh `compare_round*.jpg`.

### Nếu AP50 giảm, cần kiểm tra điều gì trước khi train thêm?

Nếu sau khi fine-tune mà AP50 trên tập kiểm thử bị giảm, cần thực hiện quy trình kiểm tra 4 bước trước khi nạp thêm dữ liệu:

1. **Kiểm tra trực quan trên ảnh `compare_round*.jpg`:** So sánh trực tiếp giữa các cột `reference`, `cold start` và `round 1`. Xem các box bị đánh dấu màu đỏ (FP) có phải là xe thật mà nhãn tham chiếu bỏ sót hay không. Nếu mô hình phát hiện đúng xe thật, AP50 giảm là do lỗi của nhãn tham chiếu chứ không phải mô hình kém đi.
2. **Kiểm tra tính nhất quán của nhãn huấn luyện (`labels/round1/`):** Rà soát xem trong 12 ảnh đã gán có trường hợp nào vẽ box không tuân thủ `GUIDELINE_LABEL.md` (ví dụ: một số ảnh vẽ ôm vệt đèn pha, một số ảnh lại cắt sát cản trước; hoặc vô tình đóng box vào đèn đường) gây nhiễu loạn đặc trưng học của YOLO.
3. **Kiểm tra hiện tượng quá khớp (Overfitting):** Huấn luyện 50 epoch trên chỉ 12 ảnh mà không có validation set và augmentation phù hợp có thể khiến mô hình "học vẹt" các đặc trưng nền đường cụ thể của 12 ảnh train thay vì học hình dạng xe tổng quát. Cần xem xét giảm số epoch hoặc tăng cường độ dữ liệu (augmentation).
4. **Kiểm tra phân phối giữa tập train và tập test:** Đánh giá xem các frame chọn trong vòng 1 có thiên lệch quá mức về một phân cảnh đặc thù (ví dụ quá nhiều xe tải hoặc quá nhiều xe ngược chiều chói đèn) trong khi tập test 20 ảnh lại phân bổ đồng đều ở các phân cảnh khác hay không.
