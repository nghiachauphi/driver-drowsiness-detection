# Cài đặt, huấn luyện và chạy ứng dụng phát hiện buồn ngủ

Tài liệu này dùng khi sao chép dự án sang một máy khác. Có hai cách sử dụng:

- Chỉ chạy Streamlit bằng mô hình đã huấn luyện: cần `app.py` và mô hình E6.
- Huấn luyện lại rồi chạy Streamlit: cần thêm notebook, script huấn luyện và dataset.

> **Kết luận nhanh:** đổi sang máy khác **không cần huấn luyện lại** nếu tệp
> `outputs/models/E6_MobileNetV2_subject.keras` đã có trong source. Máy mới chỉ
> cần tạo lại môi trường Python, cài thư viện và chạy `app.py`.

Các lệnh phải được chạy trong thư mục gốc của dự án, tức thư mục chứa `app.py`.

## 1. Những tệp cần sao chép

Nên sao chép toàn bộ thư mục dự án. Tối thiểu cần giữ cấu trúc sau:

```text
thu_muc_du_an/
├── app.py
├── DDD_Drowsiness_Executed.ipynb
├── run_local_notebook.py
├── HUONG_DAN_CAI_DAT.md
└── outputs/
    └── models/
        └── E6_MobileNetV2_subject.keras
```

Ý nghĩa của từng thành phần:

- `app.py`: giao diện Streamlit.
- `DDD_Drowsiness_Executed.ipynb`: mã nguồn thí nghiệm và huấn luyện.
- `run_local_notebook.py`: chuẩn bị notebook cho máy hiện tại, lấy dataset và chạy huấn luyện tự động.
- `outputs/models/E6_MobileNetV2_subject.keras`: mô hình mà Streamlit sử dụng.

Nếu chỉ chạy mô hình có sẵn, không được thiếu tệp E6. Nếu huấn luyện lại từ đầu, script sẽ tạo hoặc ghi đè tệp E6 này.

Không sao chép thư mục `.venv` từ máy cũ. Môi trường ảo chứa đường dẫn tuyệt đối và phải được tạo lại trên máy mới. Các tệp `python-install.log`, `streamlit-server.log`, `streamlit-server-error.log` và cache dataset cũng không bắt buộc phải sao chép.

## 2. Yêu cầu hệ thống

- Python 3.11 64-bit, là phiên bản đã được kiểm tra với dự án.
- Kết nối Internet khi cài thư viện, tải dataset và tải trọng số ImageNet lần đầu.
- Trình duyệt Chrome, Edge, Firefox hoặc Safari mới.
- Webcam chỉ bắt buộc khi dùng chế độ Webcam; vẫn có thể tải ảnh hoặc video lên.
- Nên còn đủ dung lượng cho dataset, môi trường Python, các mô hình và kết quả huấn luyện.

Ứng dụng và quá trình huấn luyện có thể chạy bằng CPU. Huấn luyện đầy đủ sáu thí nghiệm trên CPU có thể mất nhiều giờ. Nếu muốn dùng NVIDIA GPU với TensorFlow mới, nên dùng Linux hoặc WSL2 và làm theo tài liệu TensorFlow tương ứng; TensorFlow mới không hỗ trợ CUDA trực tiếp trên Windows native.

## 3. Cài Python 3.11

### Windows

Nếu dự án có sẵn `python-3.11.9-amd64.exe`, có thể chạy tệp này. Nếu không, tải Python 3.11 64-bit từ [python.org](https://www.python.org/downloads/). Khi cài, chọn **Add Python to PATH**.

Mở PowerShell mới và kiểm tra:

```powershell
py -3.11 --version
```

Nếu máy không có Python Launcher `py`, thử:

```powershell
python --version
```

### macOS

```bash
brew install python@3.11
python3.11 --version
```

Cũng có thể dùng bộ cài Python 3.11 từ python.org.

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
python3.11 --version
```

## 4. Tạo môi trường riêng và cài thư viện

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "streamlit>=1.61" "streamlit-webrtc>=0.77" "tensorflow-cpu>=2.21" "opencv-python-headless==4.10.0.84" "numpy>=1.26" pillow pandas matplotlib seaborn scikit-learn kagglehub nbformat nbclient ipykernel jupyterlab
```

Nếu `py` không tồn tại, thay lệnh đầu bằng:

```powershell
python -m venv .venv
```

Không bắt buộc kích hoạt môi trường vì các lệnh trong tài liệu gọi trực tiếp `.venv\Scripts\python.exe`.

### macOS hoặc Linux

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "streamlit>=1.61" "streamlit-webrtc>=0.77" "tensorflow-cpu>=2.21" "opencv-python-headless==4.10.0.84" "numpy>=1.26" pillow pandas matplotlib seaborn scikit-learn kagglehub nbformat nbclient ipykernel jupyterlab
```

Trên macOS, nếu không cài được `tensorflow-cpu`, thay gói đó bằng `tensorflow>=2.21`.

Kiểm tra môi trường sau khi cài:

```powershell
# Windows
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import tensorflow, cv2, streamlit, sklearn, kagglehub; print('Cài đặt thư viện thành công')"
```

```bash
# macOS/Linux
.venv/bin/python -m pip check
.venv/bin/python -c "import tensorflow, cv2, streamlit, sklearn, kagglehub; print('Cài đặt thư viện thành công')"
```

## 5. Chạy nhanh Streamlit bằng mô hình có sẵn

Trước tiên, kiểm tra tệp sau tồn tại:

```text
outputs/models/E6_MobileNetV2_subject.keras
```

Trên Windows:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Trên macOS/Linux:

```bash
.venv/bin/python -m streamlit run app.py
```

Mở `http://localhost:8501`. Để dừng ứng dụng, quay lại Terminal/PowerShell và nhấn `Ctrl + C`.

### 5.1. Đổi máy có cần huấn luyện lại không?

Không cần huấn luyện lại khi mục đích là chạy ứng dụng bằng mô hình E6 đã có. Mô hình Keras có thể được nạp trên máy khác bằng CPU; máy mới không bắt buộc phải có GPU, dataset hoặc notebook đã thực thi.

Quy trình trên máy mới:

1. Clone repository từ GitHub hoặc sao chép thư mục source đã có mô hình E6.
2. Không sao chép `.venv` từ máy cũ.
3. Cài Python 3.11, tạo `.venv` và cài thư viện theo mục 3–4.
4. Kiểm tra `outputs/models/E6_MobileNetV2_subject.keras` tồn tại.
5. Chạy Streamlit bằng lệnh ở mục 5.

Nếu lấy source từ GitHub:

```powershell
git clone https://github.com/nghiachauphi/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

Chỉ cần huấn luyện lại trong các trường hợp sau:

- Muốn cải thiện độ chính xác hoặc thay đổi ngưỡng/chiến lược huấn luyện.
- Thay dataset, cách chia dữ liệu, kiến trúc mô hình hoặc tham số thí nghiệm.
- Muốn tạo lại toàn bộ kết quả E1–E6 trên máy mới.
- Tệp E6 bị thiếu, hỏng hoặc không nạp được và không thể lấy lại từ máy cũ/GitHub.

| Mục đích trên máy mới | Có cần train lại? | Thành phần cần thiết |
|---|---|---|
| Chạy Streamlit với mô hình E6 | Không | Source, E6, Python và thư viện |
| Xem notebook/kết quả đã xuất | Không | Notebook và thư mục `outputs` |
| Thử ảnh, video hoặc webcam | Không | Source, E6 và thiết bị đầu vào tương ứng |
| Thay đổi hoặc cải thiện mô hình | Có | Notebook, script, dataset và tài nguyên huấn luyện |

## 6. Huấn luyện lại trên máy mới (không bắt buộc)

### 6.0. Pipeline cải tiến nên dùng

Pipeline mới nằm trong `train_improved_model.py`. Khác với lượt thử E1-E6 cũ, pipeline này:

- chia train/validation/test theo người bằng `StratifiedGroupKFold`, không để ảnh của cùng một người lọt sang nhiều tập;
- chọn ngưỡng cảnh báo bằng validation, tuyệt đối không chọn bằng test;
- tăng cường dữ liệu về lật ngang, xoay, zoom, độ sáng và tương phản;
- huấn luyện MobileNetV2 hai giai đoạn, có class weight, label smoothing và early stopping;
- báo cáo cả chỉ số theo ảnh và trung bình theo từng người;
- chỉ cho Streamlit dùng mô hình mới nếu lượt train đầy đủ đạt ROC AUC và balanced accuracy tối thiểu 0,55 trên test gồm những người chưa thấy khi train.

Chạy kiểm tra nhanh để phát hiện lỗi môi trường (mô hình tạo ra ở bước này không được triển khai):

```powershell
.\.venv\Scripts\python.exe train_improved_model.py --quick
```

Chạy huấn luyện đầy đủ:

```powershell
.\.venv\Scripts\python.exe train_improved_model.py
```

Nếu có bộ ảnh tự thu thập đã được đồng ý sử dụng, sắp xếp theo thư mục `Drowsy`/`Non-Drowsy` và thêm bằng:

```powershell
.\.venv\Scripts\python.exe train_improved_model.py --extra-data "D:\du_lieu\tai_xe_cua_toi"
```

Nên thu nhiều người, ánh sáng, góc đầu, kính, loại camera và trạng thái thật. Không nên chỉ tăng số khung hình gần như giống nhau từ một vài video vì điều đó làm chỉ số theo ảnh cao giả tạo mà không cải thiện khả năng nhận người mới.

Kết quả mới được ghi vào `outputs/models/improved_mobilenetv2.keras`, `outputs/results/improved_model_metadata.json` và `outputs/tables/improved_subject_metrics.csv`. Khởi động lại Streamlit sau khi train để nạp lại model. Nếu mô hình không vượt cổng chất lượng, ứng dụng tự giữ E6 và kết hợp kiểm tra mắt cùng suy luận theo thời gian.

Script sử dụng dataset Kaggle `ismailnasri20/driver-drowsiness-dataset-ddd`. Nếu không truyền đường dẫn dataset, KaggleHub sẽ tải dataset công khai vào cache của người dùng hoặc dùng lại bản đã tải trước đó.

### 6.1. Chạy thử pilot trước

Pilot dùng 2% dữ liệu để kiểm tra toàn bộ pipeline. Vì source đã chứa kết quả cũ, dùng `--force` để thực sự huấn luyện lại:

```powershell
# Windows
.\.venv\Scripts\python.exe run_local_notebook.py --force
```

```bash
# macOS/Linux
.venv/bin/python run_local_notebook.py --force
```

### 6.2. Huấn luyện đầy đủ

Sau khi pilot chạy thành công:

```powershell
# Windows
.\.venv\Scripts\python.exe run_local_notebook.py --full --force
```

```bash
# macOS/Linux
.venv/bin/python run_local_notebook.py --full --force
```

`--full` sử dụng toàn bộ dataset. `--force` cho phép ghi đè các kết quả thí nghiệm E1–E6 đã có trong `outputs`. Nếu cần giữ kết quả cũ, hãy sao lưu thư mục `outputs` trước.

### 6.3. Dùng dataset đã tải thủ công

Có thể bỏ qua việc tải bằng KaggleHub và truyền thư mục dataset đã giải nén:

```powershell
# Windows
.\.venv\Scripts\python.exe run_local_notebook.py --dataset "D:\du_lieu\driver-drowsiness-dataset-ddd" --full --force
```

```bash
# macOS/Linux
.venv/bin/python run_local_notebook.py --dataset "/duong-dan/driver-drowsiness-dataset-ddd" --full --force
```

Script tìm ảnh đệ quy trong thư mục được truyền. Cấu trúc dataset phải còn các thư mục lớp có tên chứa `Drowsy`; lớp không buồn ngủ phải có `Non` hoặc `Not` trong tên thư mục.

Dataset công khai thường tải được mà không đăng nhập. Nếu Kaggle yêu cầu xác thực hoặc chấp nhận điều khoản, tạo API token trong tài khoản Kaggle rồi làm theo hướng dẫn xác thực của [KaggleHub](https://github.com/Kaggle/kagglehub#authenticate). Không ghi token vào source hoặc tài liệu của dự án.

### 6.4. Kết quả huấn luyện

Trong khi chạy, script tạo `DDD_Drowsiness_Local.ipynb` và lưu kết quả tại:

```text
outputs/
├── figures/    # biểu đồ
├── models/     # mô hình E1-E6
├── results/    # lịch sử và chỉ số
└── tables/     # bảng CSV
```

Mô hình dùng cho ứng dụng là:

```text
outputs/models/E6_MobileNetV2_subject.keras
```

Sau khi huấn luyện xong, dừng Streamlit cũ nếu đang chạy rồi chạy lại lệnh ở mục 5 để ứng dụng nạp mô hình mới.

## 7. Mở notebook để xem kết quả

Sau khi script đã tạo notebook local:

```powershell
# Windows
.\.venv\Scripts\python.exe -m jupyter lab DDD_Drowsiness_Local.ipynb
```

```bash
# macOS/Linux
.venv/bin/python -m jupyter lab DDD_Drowsiness_Local.ipynb
```

Không sửa đường dẫn máy cũ trong `DDD_Drowsiness_Executed.ipynb`. Script sẽ tạo bản `DDD_Drowsiness_Local.ipynb` có đường dẫn đúng với máy hiện tại.

## 8. Sử dụng Streamlit

1. Chọn ngưỡng cảnh báo; mặc định là 0,50.
2. Chọn **Webcam** hoặc **Tải ảnh/video**.
3. Với webcam, nhấn **START** và cấp quyền camera cho `localhost`.
4. Với tệp tải lên, dùng ảnh `JPG/PNG/BMP` hoặc video `MP4/AVI/MOV/MKV`, tối đa 200 MB.
5. Với video, chọn tần suất phân tích, khoảng cách giữa hai ảnh cảnh báo và số ảnh tối đa; sau đó nhấn **Phân tích video và tách ảnh cảnh báo**.
6. Ứng dụng đặt video gốc và video nhận dạng có khung mặt/xác suất cạnh nhau để so sánh, đồng thời hiển thị đồ thị xác suất theo thời gian.
7. Các khung hình vượt ngưỡng được tách riêng thành ảnh JPEG; có thể tải video nhận dạng dưới dạng MP4.
8. Có thể tải một tệp ZIP chứa toàn bộ ảnh cảnh báo và báo cáo CSV gồm thời điểm, số khung hình, xác suất và ngưỡng.
9. Khung xanh là *Tỉnh táo*; khung đỏ là *Cảnh báo buồn ngủ* khi xác suất lớn hơn hoặc bằng ngưỡng.

## 9. Xử lý lỗi thường gặp

| Lỗi/hiện tượng | Cách xử lý |
|---|---|
| Không nhận `py`, `python` hoặc `python3.11` | Cài Python 3.11 64-bit, chọn Add Python to PATH và mở Terminal mới. |
| `.venv` được sao chép từ máy cũ nhưng không chạy | Xóa bản `.venv` đã sao chép và tạo lại theo mục 4. |
| Không tìm thấy `DDD_Drowsiness_Executed.ipynb` | Sao chép lại notebook này vào cùng thư mục với `run_local_notebook.py`. |
| Notebook vẫn hiện kết quả cũ và không huấn luyện | Chạy script với `--force`. |
| KaggleHub không tải được dataset | Kiểm tra Internet; đăng nhập/chấp nhận điều khoản dataset nếu được yêu cầu, hoặc tải thủ công và dùng `--dataset`. |
| Không tìm thấy ảnh trong dataset | Truyền đúng thư mục gốc đã giải nén và giữ nguyên tên các thư mục `Drowsy`/`Non-Drowsy`. |
| Không tải được trọng số ImageNet | Kiểm tra Internet hoặc proxy/firewall; lần xây dựng ResNet50/MobileNetV2 đầu tiên cần tải trọng số. |
| TensorFlow hết RAM | Dùng pilot trước; giảm `batch_size` trong notebook local hoặc dùng máy nhiều RAM hơn. |
| Streamlit không tìm thấy model | Kiểm tra `outputs/models/E6_MobileNetV2_subject.keras` và chạy lệnh tại thư mục chứa `app.py`. |
| Cổng 8501 đang được dùng | Thêm `--server.port 8502` vào cuối lệnh Streamlit. |
| Không thấy camera | Cấp quyền camera cho `localhost` và đóng Zoom/Teams hoặc ứng dụng khác đang dùng webcam. |
| `NotFoundError: Requested device not found` | Trình duyệt không thấy thiết bị video. Kết nối/mở công tắc webcam, bật camera bằng phím chức năng của laptop, kiểm tra **Device Manager > Cameras**, rồi tải lại trang. Nếu vẫn lỗi, dùng chế độ **Tải ảnh/video**. |
| `Connection is taking longer than expected` | Streamlit Cloud không vượt qua được NAT/firewall chỉ bằng STUN. Thêm `TURN_URLS`, `TURN_USERNAME` và `TURN_CREDENTIAL` của Metered/coturn vào **Manage app > Settings > Secrets**, lưu lại, reboot app rồi thử lại webcam. Tài khoản Twilio Trial không cấp TURN/NTS. |

## 10. Kiểm tra trước khi chép sang máy khác

- Có `app.py`, `DDD_Drowsiness_Executed.ipynb`, `run_local_notebook.py` và tài liệu này.
- Có E6 nếu muốn chạy Streamlit ngay mà không huấn luyện.
- Không đóng gói `.venv`, cache dataset hoặc các tệp log.
- Trên máy mới, tạo lại `.venv` và cài lại thư viện.
- Nếu chỉ chạy ứng dụng, không cần dataset và không chạy lệnh huấn luyện.
- Chỉ khi cần huấn luyện lại: chạy pilot với `--force`, sau đó mới chạy `--full --force`.
- Chạy Streamlit và kiểm tra ảnh/video trước khi thử webcam.

Mô hình hiện phục vụ đồ án và minh họa học thuật, không phải thiết bị cảnh báo giao thông, chẩn đoán y tế hoặc hệ thống an toàn thực tế.
