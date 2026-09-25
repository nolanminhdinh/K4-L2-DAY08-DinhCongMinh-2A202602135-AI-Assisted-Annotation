# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, nếu chỉ có ngân sách rà soát 5 ảnh, tôi sẽ ưu tiên chọn 5 frame sau:
1. `frame_0182.jpg` (rank 1, score 0.9591, t=72.8s): Frame có điểm bất định cao nhất toàn bộ pool, độ mập mờ đạt đỉnh (A=1.0, 18 box có conf 0.15–0.50), mật độ xe dày đặc với nhiều tình huống xe chạy ngược chiều và cùng chiều có độ bất định cao.
2. `frame_0369.jpg` (rank 2, score 0.9324, t=147.6s): Đại diện cho đoạn cuối video với 43 box phát hiện, U rất cao (0.9315), phân tán đa dạng các làn xe.
3. `frame_0099.jpg` (rank 8, score 0.9063, t=39.6s): Nằm ở khoảng thời gian đầu video (t=39.6s, cách xa các frame trên > 30s), U đạt tới 0.9460, giúp rà soát các trường hợp xe ngược chiều có thân xe tối.
4. `frame_0227.jpg` (rank 11, score 0.8915, t=90.8s): Frame ở đoạn giữa video (t=90.8s), xuất hiện phương tiện kích thước lớn đặc thù là xe tải thùng (box truck) gây phân vân mạnh cho mô hình (A=0.7778, 14 box mập mờ).
5. `frame_0270.jpg` (rank 13, score 0.8878, t=108.0s): Bổ sung vùng thời gian t=108.0s, cách đều giữa frame 227 và 369, giúp trải đều mẫu theo thời gian và tránh việc tập trung quá nhiều ảnh vào cụm giây 130–150.
Quyết định này chủ động loại bỏ các frame có điểm rất cao như `frame_0372.jpg` (rank 6, score 0.9101, t=148.8s) do chỉ cách `frame_0369.jpg` 1.2s (< MIN_GAP_S = 2.0s), bối cảnh gần như trùng lặp hoàn toàn, gây lãng phí ngân sách gán nhãn mà không mang lại tri thức mới.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:
- `frame_0182.jpg` (rank 1, score=0.9591, U=0.9182, A=1.0000, 28 box, 18 box mơ hồ): Là frame có số lượng box mơ hồ lớn nhất pool (A=1.0), trên contact sheet thấy rõ nhiều xe ở cự ly trung bình với đèn pha gây chói.
- `frame_0331.jpg` (rank 5, score=0.9154, U=0.8308, A=1.0000, 47 box, 18 box mơ hồ): Là frame có lượng box nhiều nhất trong lô (47 box), mật độ giao thông nghẽn đặc, model gặp khó khăn trong việc tách các xe đi sát nhau.
- `frame_0099.jpg` (rank 8, score=0.9063, U=0.9460, A=0.7778, 29 box, 14 box mơ hồ): Có U cao vượt trội (0.9460), các box mập mờ tập trung ở làn xe đi tới với thân xe chìm trong bóng tối.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
- Frame điểm cao nhưng không chọn: `frame_0372.jpg` (rank 6, score=0.9101, U=0.9202, A=0.8333). Mặc dù đứng thứ 6 trong toàn bộ pool 268 ảnh, frame này bị thuật toán tham lam loại bỏ vì thời điểm t=148.8s quá gần `frame_0369.jpg` (t=147.6s, cách 1.2s < MIN_GAP_S 2.0s). Bối cảnh và vị trí của các xe gần như y hệt `frame_0369.jpg`, việc gán nhãn cả hai sẽ gây trùng lặp và lãng phí công sức.
- Frame điểm thấp hơn vẫn nên xem: `frame_0002.jpg` (rank 18, score=0.8658, t=0.8s) hoặc `frame_0020.jpg` (rank 29, score=0.8451, t=8.0s): Nằm ở đầu video với điều kiện mật độ giao thông thưa hơn, giúp mô hình học các mẫu xe chạy tốc độ cao hoặc xe đơn lẻ không bị che khuất.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:
- Điểm bất định cao (U, A) chỉ phản ánh sự phân vân của mô hình tại thời điểm hiện tại (confidence gần 0.5 hoặc nhiều box trong dải 0.15–0.50); nó KHÔNG chứng minh hay bảo đảm rằng khi gắn nhãn các frame này thì mô hình fine-tune sẽ tăng AP50 trên tập test. Điểm bất định cao có thể do nhiễu môi trường (vệt đèn pha phản chiếu trên mặt đường ướt, ánh đèn cầu vượt, biển báo phản quang) khiến mô hình bối rối, tức là thông tin khó nhưng là nhiễu không có giá trị học. Hơn nữa, việc cải thiện số đo trên tập test còn phụ thuộc vào mức độ tương đồng giữa các ca khó trong pool với phân phối của 20 ảnh test và nguy cơ overfit trên tập dữ liệu nhỏ (12 ảnh).
