# Quét độc lập trước khi xem pre-label

Frame: frame_0182.jpg

Số xe nhìn thấy bằng mắt: 24

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:
1. Mép dưới cùng giữa ảnh (gần camera nhất, toạ độ khoảng cx 0.43, cy 0.94): Thân xe con màu tối đang tiến vào khung hình từ mép dưới, bị cắt mép chỉ thấy nóc capo và kính lái, không có đèn rọi thẳng vào camera nên AI rất dễ bỏ sót hoàn toàn (FN).
2. Làn giữa có xe đang chạy tới (toạ độ khoảng cx 0.53, cy 0.73): Đèn pha chiếu vệt sáng dài chói lòa xuống mặt đường bê tông; AI rất dễ bị đánh lừa và kéo box xuống quá thấp bao trọn cả vệt sáng phản chiếu thay vì ôm sát thân xe.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
