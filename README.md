# VisionDrive AI - Cảnh Báo Lái Xe Theo Loại Đường Giao Thông
> **Đề tài 6: Notification for Driver based on type of road**  
> *Hệ thống hỗ trợ lái xe an toàn (ADAS) nhận diện loại đường bằng Computer Vision và điều chỉnh cảnh báo tốc độ theo Thông tư 31/2019/TT-BGTVT.*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-Export_Ready-005CED.svg?style=flat&logo=onnx)](https://onnx.ai)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-12%20Passed%20(100%25)-success.svg)](#kiểm-thử-hệ-thống)

---

## 📌 Giới thiệu Đề tài

Trong bối cảnh giao thông tại Việt Nam, các quy định về tốc độ tối đa cho phép phụ thuộc chặt chẽ vào **loại hình đường giao thông** và **khu vực** (trong hay ngoài khu đông dân cư, đường đôi có dải phân cách giữa hay đường hai chiều, đường cao tốc hay ngõ hẻm). Người tham gia giao thông thường gặp khó khăn trong việc nhận biết và ghi nhớ chính xác tốc độ giới hạn tại các đoạn đường không cắm biển lặp lại.

**VisionDrive AI** giải quyết bài toán trên bằng cách:
1. **Phân loại thời gian thực (Real-time Vision)**: Nhận diện 5 loại đường giao thông điển hình tại Việt Nam từ camera hành trình bằng mô hình MobileNetV3 tối ưu hóa.
2. **Bộ lọc làm mượt thời gian (Temporal Smoothing)**: Sử dụng giải thuật Exponential Moving Average (EMA) cùng cơ chế trễ (hysteresis) giúp nhãn dự đoán ổn định qua từng khung hình video, loại bỏ hiện tượng nhấp nháy (flickering).
3. **Động cơ cảnh báo tốc độ 4 trạng thái (4-State Alert Engine)**: So sánh tốc độ GPS thực tế của xe với quy chuẩn tốc độ theo **Thông tư 31/2019/TT-BGTVT**, đưa ra cảnh báo giải thích minh bạch ("Explainable AI").
4. **Hệ thống phản hồi từ người dùng (Feedback Loop)**: Hỗ trợ người dùng "Báo sai" khi hệ thống nhận diện nhầm, ghi nhận dữ liệu phục vụ tái huấn luyện (Active Learning).

---

## 🎯 5 Loại Đường Giao Thông Việt Nam

Hệ thống phân loại đường theo 5 nhóm đặc trưng tại Việt Nam:

| Mã Class | Tên tiếng Việt | Giới hạn Ô tô | Giới hạn Xe máy | Đặc điểm nhận diện trực quan |
| :--- | :--- | :---: | :---: | :--- |
| `cao_toc` | **Cao tốc** | 100 - 120 km/h | Cấm xe máy | Nhiều làn rộng, dải phân cách cứng, rào chắn hộ lan, mặt đường nhựa tiêu chuẩn cao |
| `duong_doi_do_thi` | **Đường đôi đô thị** | 60 km/h | 60 km/h | Trong khu dân cư, có dải phân cách cứng ở giữa chia hai chiều riêng biệt |
| `pho_hai_chieu` | **Phố hai chiều** | 50 km/h | 50 km/h | Trong khu dân cư, không có dải phân cách giữa (vạch kẻ liền hoặc nét đứt) |
| `ngoai_khu_dan_cu`| **Ngoài khu dân cư** | 80 - 90 km/h | 70 km/h | Đường quốc lộ, tỉnh lộ thoáng đãng, mật độ nhà dân thấp |
| `hem_ngo` | **Hẻm / ngõ** | 30 km/h | 30 km/h | Không gian hẹp, sát tường nhà, không có vạch phân làn, nhiều góc khuất |

---

## 🔔 Bốn Trạng Thái Cảnh Báo (4-State Alert Engine)

Hệ thống phân cấp cảnh báo thành **4 trạng thái trực quan**:

```
                  ┌─────────────────────────────────────┐
                  │ 1. ĐANG XÁC ĐỊNH (idle)            │
                  │ Chưa đủ khung hình hoặc Conf < 55%  │
                  └──────────────────┬──────────────────┘
                                     │ Đủ Warmup & Conf >= 55%
                  ┌──────────────────▼──────────────────┐
                  │ Phân loại đường: Giới hạn V_max     │
                  └──────────────────┬──────────────────┘
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
  v <= V_max - margin     V_max - margin < v <= V_max      v > V_max
            │                        │                        │
┌───────────▼───────────┐  ┌─────────▼─────────┐  ┌───────────▼───────────┐
│ 2. AN TOÀN (ok)       │  │ 3. CHÚ Ý (warn)   │  │ 4. VƯỢT TỐC (bad)     │
│ Vận tốc trong giới hạn│  │ Sắp chạm giới hạn │  │ Vượt tốc độ quy định  │
│ (Màu xanh lục)        │  │ (Màu vàng hổ phách│  │ (Màu đỏ + Âm thanh)   │
└───────────────────────┘  └───────────────────┘  └───────────────────────┘
```

1. **Đang xác định (`idle`)**:
   - Khi xe mới khởi động hoặc đang chuyển cảnh đột ngột, chưa thu thập đủ số khung hình khởi động (mặc định 3 frames) hoặc độ tin cậy mô hình chưa đạt ngưỡng tin cậy tối thiểu (mặc định 0.55).
   - Hệ thống giữ trạng thái chờ an toàn, không cảnh báo vội vã.
2. **An toàn (`ok`)**:
   - Tốc độ xe hiện tại nằm hoàn toàn trong giới hạn quy định của đoạn đường.
3. **Chú ý (`warn`)**:
   - Tốc độ xe tiến gần đến giới hạn cho phép (nằm trong dải `[Giới hạn - Margin, Giới hạn]`, mặc định `margin = 5 km/h`, có thể tùy chỉnh linh hoạt qua Cài đặt).
   - Báo cho tài xế chủ động nhả ga hoặc rà phanh nhẹ.
4. **Vượt tốc (`bad`)**:
   - Tốc độ xe vượt quá giới hạn tối đa cho phép của đoạn đường.
   - Kích hoạt cảnh báo cường độ cao kèm còi/âm thanh và câu giải thích chi tiết.

### ✨ Mỗi Cảnh Báo Đều Có Lời Giải Thích (Explainable AI)
Hệ thống tạo câu giải thích tự động theo thời gian thực giúp tài xế luôn hiểu rõ lý do cảnh báo:
- *Ví dụ:* `"Đang ở Phố hai chiều (độ tin cậy 92%), giới hạn 50 km/h. Tốc độ hiện tại 54 km/h (vượt 4 km/h)."`

---

## 🏗️ Cấu Trúc Dự Án (Repository Architecture)

```text
.
├── backend/                        # Backend REST API (FastAPI)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI Application, CORS, Lifespan
│   │   ├── schemas.py              # Pydantic Schemas (Request/Response)
│   │   ├── routes/
│   │   │   ├── predict.py          # Suy luận Frame, Ảnh, Video, Demo Scenarios
│   │   │   ├── rules.py            # Quản lý & tra cứu luật tốc độ TT 31/2019
│   │   │   └── feedback.py         # Tiếp nhận phản hồi "Báo sai" từ người dùng
│   │   └── services/
│   │       ├── classifier.py       # MobileNetV3 Road Classifier (GPU/CPU)
│   │       ├── smoothing.py        # Exponential Moving Average (EMA) Smoother
│   │       └── warning.py          # Động cơ 4 trạng thái cảnh báo & giải thích
│   ├── data/
│   │   └── feedback_flags.json     # Lưu trữ dữ liệu cắm cờ báo sai
│   ├── models/
│   │   └── vietnam_road_classifier.pt # Trọng số mô hình đã huấn luyện
│   ├── rules/
│   │   └── speed_rules.json        # Dữ liệu quy chuẩn tốc độ Việt Nam
│   ├── tests/
│   │   ├── conftest.py
│   │   └── test_warning.py         # 12 bài kiểm thử tự động toàn diện
│   ├── Dockerfile                  # Container hóa Backend
│   ├── pytest.ini
│   └── requirements.txt
├── ml/                             # Machine Learning Pipeline
│   ├── train.py                    # Script huấn luyện MobileNetV3 5 classes
│   ├── evaluate.py                 # Đánh giá Confusion Matrix, Macro F1
│   ├── export_onnx.py              # Xuất mô hình sang định dạng ONNX
│   └── benchmark.py                # Đo kiểm Latency (ms) và FPS trên GPU/CPU
├── models/                         # Lưu trữ mô hình xuất bản
│   ├── vietnam_road_classifier.pt  # PyTorch Checkpoint (4.13 MB)
│   ├── vietnam_road_classifier.onnx# ONNX Runtime Model (4.01 MB)
│   └── road_classifier_eval.json   # Kết quả đánh giá mô hình
├── frontend/                       # Giao diện người dùng
│   ├── design/
│   │   ├── Main.dc.html            # Bản thiết kế giao diện ADAS Dashboard
│   │   ├── Landing.dc.html         # Bản thiết kế giao diện Landing Page
│   │   ├── support.js              # Mã JavaScript điều khiển tương tác
│   │   └── canvas.json
│   └── lib/
│       ├── api.ts                  # Client gọi API Backend FastAPI
│       ├── types.ts                # TypeScript interfaces chuẩn hóa
│       └── mock.ts                 # Dữ liệu demo fallback
├── data/
│   ├── labels.csv                  # Danh mục nhãn dữ liệu chuẩn
│   └── README.md                   # Hướng dẫn thu thập & tiền xử lý dataset
├── .env.example                    # Mẫu biến môi trường cấu hình
├── .gitignore                      # Cấu hình bỏ qua file rác và video lớn
├── docker-compose.yml              # Khởi chạy toàn bộ hệ thống bằng Docker
└── README.md                       # Tài liệu đồ án
```

---

## ⚡ Hiệu Năng Mô Hình & Đo Kiểm (Benchmark)

Mô hình được tối ưu trên kiến trúc **MobileNetV3-Small** với phân loại 5 lớp đường, đạt tốc độ suy luận vượt trội phù hợp với các thiết bị biên (Edge Devices / Jetson Nano / Raspberry Pi 5 / Dashcam):

- **Kích thước mô hình:** 4.13 MB (`.pt`) / 4.01 MB (`.onnx`)
- **Phần cứng đo kiểm:** NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.8)
- **Độ trễ trung bình (Average Latency):** **11.69 ms / frame**
- **Tốc độ xử lý (Throughput):** **85.54 FPS** (đáp ứng xuất sắc thời gian thực > 30 FPS)
- **Độ chính xác kiểm thử (Validation Accuracy):** **98.7%**

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Chuẩn bị môi trường Python

Khuyến nghị sử dụng Python 3.10, 3.11 hoặc 3.12:

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Kích hoạt môi trường (Linux/macOS)
source .venv/bin/activate

# Cài đặt thư viện phụ thuộc
pip install -r backend/requirements.txt
```

### 2. Khởi chạy Backend API

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Hệ thống sẽ chạy tại địa chỉ:
- **API Server:** `http://localhost:8000`
- **Swagger UI Interactive Docs:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## 📡 Danh Sách API Endpoints

### 1. Nhận diện & Cảnh báo (`/api/predict`)

| Phương thức | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `POST` | `/api/predict/frame` | Nhận diện khung hình trực tiếp (Base64), trả về loại đường, 4 trạng thái cảnh báo, hộp đối tượng và câu giải thích |
| `POST` | `/api/predict/image` | Tải lên file ảnh (multipart/form-data), phân tích và trả về cảnh báo |
| `POST` | `/api/predict/video` | Tải lên đoạn video (mp4, avi), phân tích mẫu khung hình và trả về biểu đồ phân loại theo thời gian |
| `POST` | `/api/predict/reset-stream`| Đặt lại trạng thái bộ lọc làm mượt (EMA) khi bắt đầu hành trình mới |
| `GET` | `/api/predict/demo-scenarios`| Lấy 5 kịch bản mẫu phục vụ trình diễn trên Dashboard |

#### Ví dụ gọi API `/api/predict/frame`:
```bash
curl -X POST "http://localhost:8000/api/predict/frame" \
     -H "Content-Type: application/json" \
     -d '{
       "frame_base64": "<chuỗi_base64_ảnh>",
       "speed_kmh": 68.5,
       "vehicle_type": "car",
       "caution_margin_kmh": 5.0
     }'
```

#### Phản hồi mẫu từ API:
```json
{
  "road_type": "duong_doi_do_thi",
  "road_type_vi": "Đường đôi đô thị",
  "confidence": 0.942,
  "smoothed_confidence": 0.928,
  "alert_state": "bad",
  "alert_state_vi": "Vượt tốc",
  "speed_limit_kmh": 60,
  "speed_kmh": 68.5,
  "is_overspeed": true,
  "explanation": "Đang ở Đường đôi đô thị (độ tin cậy 93%), giới hạn 60 km/h. Tốc độ hiện tại 68.5 km/h (vượt 8.5 km/h).",
  "boxes": [
    {"x": 322, "y": 240, "w": 62, "h": 44, "label": "ô tô 94%", "confidence": 0.94}
  ],
  "latency_ms": 11.8
}
```

---

### 2. Quản lý Quy Chuẩn Tốc Độ (`/api/rules`)

- `GET /api/rules`: Xem toàn bộ ma trận tốc độ theo Thông tư 31/2019/TT-BGTVT.
- `GET /api/rules/{road_type}`: Xem chi tiết quy định của một loại đường cụ thể.
- `PUT /api/rules/{road_type}`: Cho phép cập nhật hoặc tinh chỉnh lại giới hạn tốc độ (ví dụ khi đoạn đường đang thi công).

---

### 3. Tiếp nhận Phản hồi & Báo Sai (`/api/feedback`)

- `POST /api/feedback`: Lưu cờ phản hồi khi mô hình nhận diện sai (kèm nhãn thực tế của người dùng).
- `GET /api/feedback`: Xem danh sách tất cả các trường hợp bị người dùng báo sai để thu thập thêm mẫu huấn luyện.
- `DELETE /api/feedback`: Xóa lịch sử phản hồi.

---

## 🧪 Kiểm Thử Hệ Thống (Unit & Integration Tests)

Hệ thống đi kèm bộ kiểm thử tự động 100% bao phủ toàn bộ 4 trạng thái cảnh báo, giải thuật lọc EMA, và các API endpoints:

```bash
# Chạy toàn bộ test suite
pytest backend/tests -v
```

**Kết quả kiểm thử:**
```text
repo_upstream/backend/tests/test_warning.py::test_warning_state_1_idle_warmup PASSED
repo_upstream/backend/tests/test_warning.py::test_warning_state_1_idle_low_confidence PASSED
repo_upstream/backend/tests/test_warning.py::test_warning_state_2_ok_safe PASSED
repo_upstream/backend/tests/test_warning.py::test_warning_state_3_warn_caution PASSED
repo_upstream/backend/tests/test_warning.py::test_warning_state_4_bad_overspeed PASSED
repo_upstream/backend/tests/test_warning.py::test_custom_caution_margin PASSED
repo_upstream/backend/tests/test_warning.py::test_temporal_smoother PASSED
repo_upstream/backend/tests/test_warning.py::test_api_health PASSED
repo_upstream/backend/tests/test_warning.py::test_api_rules PASSED
repo_upstream/backend/tests/test_warning.py::test_api_demo_scenarios PASSED
repo_upstream/backend/tests/test_warning.py::test_api_predict_frame PASSED
repo_upstream/backend/tests/test_warning.py::test_api_feedback_flow PASSED

======================== 12 passed in 8.15s (100%) ========================
```

---

## 🛠️ Công Cụ Machine Learning & Xuất Mô Hình

```bash
# 1. Đánh giá mô hình và sinh ma trận nhầm lẫn (Confusion Matrix)
python ml/evaluate.py --checkpoint models/vietnam_road_classifier.pt

# 2. Đo kiểm FPS và độ trễ trên GPU
python ml/benchmark.py --device cuda --runs 200

# 3. Xuất mô hình sang ONNX tối ưu
python ml/export_onnx.py --input-pt models/vietnam_road_classifier.pt --output-onnx models/vietnam_road_classifier.onnx
```

---

## 🐳 Khởi Chạy Với Docker

```bash
# Build và chạy backend container
docker-compose up --build -d

# Xem log hoạt động
docker-compose logs -f backend
```

---

## 👥 Thành Viên Thực Hiện Đề Tài

- **Đề tài:** Đề tài 6 - Notification for Driver based on type of road
- **Hệ thống:** VisionDrive AI
- **Giảng viên hướng dẫn:** Bộ môn Khoa học Máy tính / Trí tuệ nhân tạo
