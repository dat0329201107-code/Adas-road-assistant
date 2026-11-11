# Dữ Liệu & Nhãn Phân Loại Đường Việt Nam — VisionDrive AI

## 1. Giới thiệu tập dữ liệu
Tập dữ liệu phục vụ nghiên cứu và huấn luyện mô hình nhận diện 5 loại đường giao thông đặc trưng tại Việt Nam, thu thập từ các video hành trình (dashcam) 100% thực tế được ghi hình tại Hà Nội, TP. Hồ Chí Minh và các tuyến Quốc lộ, Cao tốc Việt Nam.

Tất cả video đều được lọc chuyển động chủ động (`MIN_MOTION_DIFF >= 12.0`) nhằm loại bỏ các khung hình xe dừng đèn đỏ hoặc đỗ xe tại chỗ, đảm bảo xe đang di chuyển liên tục trên tuyến đường.

---

## 2. Danh mục 5 phân loại đường (Classes)

| ID | Mã lớp (`class_key`) | Tên tiếng Việt | Giới hạn tốc độ | Đặc điểm nhận dạng mặt đường & Cảnh quan |
| :---: | :--- | :--- | :---: | :--- |
| `0` | `cao_toc` | **Cao tốc** | **100 km/h** *(đến 120)* | Mặt đường 3-4 làn xe, dải phân cách bê tông khối cứng (Jersey barrier), hộ lan kim loại, biển báo gantry xanh trên cao, không có giao cắt đồng mức. |
| `1` | `duong_doi_do_thi` | **Đường đôi đô thị** | **60 km/h** | Đường trong đô thị có dải phân cách giữa (dải phân cách cứng hoặc bồn cây trồng hoa), 2 chiều xe chạy riêng biệt. |
| `2` | `pho_hai_chieu` | **Phố hai chiều** | **50 km/h** | Đường phố nội đô không có dải phân cách giữa, xe hai chiều lưu thông chung một mặt đường hoặc đường 1 chiều 1 làn xe cơ giới. |
| `3` | `ngoai_khu_dan_cu` | **Ngoài khu dân cư** | **80 km/h** *(đến 90)* | Quốc lộ hoặc đường liên tỉnh chạy qua đồng ruộng, đồi núi, nhà dân thưa thớt. |
| `4` | `hem_ngo` | **Hẻm / ngõ** | **30 km/h** *(an toàn)* | Ngõ ngách hẹp trong khu dân cư xe máy, tường nhà sát mặt đường, tầm nhìn hẹp, đông người đi bộ và trẻ em. |

---

## 3. Cấu trúc thư mục dữ liệu

```
data/
├── labels.csv                    # Bảng tra cứu nhãn, tốc độ và căn cứ pháp lý
├── feedback_flags.json           # Dữ liệu người dùng báo sai (Feedback & Active Learning)
└── videos/                       # 3,000 video clips đã cắt lọc (mỗi loại 600 clip)
    ├── cao_toc/                  # 600 clips
    ├── duong_doi_do_thi/         # 600 clips
    ├── pho_hai_chieu/            # 600 clips
    ├── ngoai_khu_dan_cu/         # 600 clips
    └── hem_ngo/                  # 600 clips
```

---

## 4. Căn cứ pháp lý
- Quy định về tốc độ và khoảng cách an toàn căn cứ theo **Thông tư 31/2019/TT-BGTVT** ban hành ngày 29/08/2019 của Bộ Giao thông Vận tải.
