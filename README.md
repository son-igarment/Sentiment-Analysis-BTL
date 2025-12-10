# NLP_BTL
# Phân tích cảm xúc VLSP (TextCNN/BiLSTM/CNN+BiLSTM)
Thông tin liên lạc
Email:plnson.sdh242@hcmut.edu.vn
Số điện thoại/ Zalo: 0334229371 
Notebook `Sentiment Analysis Final.ipynb` tổng hợp Lab 5 và Lab 6 để so sánh 3 mô hình phân loại cảm xúc tiếng Việt (TextCNN, BiLSTM, CNN+BiLSTM) trên bộ dữ liệu VLSP. Dưới đây là hướng dẫn chuẩn bị môi trường, dữ liệu và cách chạy.

## Cấu trúc thư mục quan trọng
- `Sentiment Analysis Final.ipynb`: notebook chính để huấn luyện và đánh giá.
- `Sentiment Analysis Final.html`: bản export để xem nhanh không cần chạy.
- `Lab 5/vlsp_sentiment_train.csv`, `Lab 5/vlsp_sentiment_test.csv`: dữ liệu train/test (tab-separated, cột `Class`, `Data`).
- `Lab 5/vi-model-CBOW.txt`: embedding Word2Vec 400 chiều (khoảng 700MB).

## Yêu cầu môi trường
- Python 3.9+ (khuyến nghị 64-bit).
- Thư viện: `pyvi`, `gensim`, `scikit-learn`, `tensorflow==2.15`, `pandas`, `numpy`, `jupyter` (nếu chạy local).
- RAM ≥ 2GB (để nạp embedding) và tốt hơn nếu có GPU cho TensorFlow.

Cài nhanh (virtualenv/conda tùy bạn):
```bash
pip install -U pyvi gensim scikit-learn tensorflow==2.15 pandas numpy jupyter
```

## Chuẩn bị đường dẫn dữ liệu
Trong notebook gốc, `BASE_DIR` đang trỏ tới Google Drive (`/content/drive/...`). Khi chạy trên máy cá nhân, sửa lại các dòng sau cho trỏ vào repo:

```python
from pathlib import Path
BASE_DIR = Path.cwd()  # hoặc Path('C:/Users/sonpln/Desktop/NLP_BTL')
TRAIN_PATH = BASE_DIR / 'Lab 5/vlsp_sentiment_train.csv'
TEST_PATH  = BASE_DIR / 'Lab 5/vlsp_sentiment_test.csv'
W2V_PATH   = BASE_DIR / 'Lab 5/vi-model-CBOW.txt'
```

Embedding đang ở dạng text Word2Vec, vì vậy trong hàm `build_embedding_matrix` hãy đặt `binary=False`:
```python
w2v = KeyedVectors.load_word2vec_format(str(W2V_PATH), binary=False)
```

Nếu chạy trên Colab, giữ nguyên `drive.mount` và đặt `BASE_DIR` tới thư mục chứa các file trên Drive.

## Quy trình tiền xử lý & mô hình (tóm tắt từ notebook)
- Tiền xử lý: loại số/khoảng trắng thừa (`clean_text`), tokenize tiếng Việt bằng `ViTokenizer`, giới hạn vocab 12k, padding độ dài 180.
- Nhãn: map {-1, 0, 1} thành one-hot 3 lớp.
- Embedding: nạp Word2Vec 400 chiều từ `vi-model-CBOW.txt`; nếu không có file sẽ random.
- Mô hình cơ bản:
  - TextCNN: 3 nhánh kernel 3/4/5 + GlobalMaxPooling.
  - BiLSTM: LSTM hai chiều 128 hidden.
  - CNN+BiLSTM: Conv1D 5, MaxPool, BiLSTM 96.
- Huấn luyện: EarlyStopping (patience=2), batch 256, 6 epoch; tách train/val 90/10 có stratify.
- Phiên bản v2 (cuối notebook) thêm class_weight, ReduceLROnPlateau và cho phép fine-tune embedding; tính thêm `macro_f1`.

## Cách chạy notebook
1. Mở notebook: `jupyter notebook` rồi chọn `Sentiment Analysis Final.ipynb` (hoặc mở bằng VS Code, Colab).
2. Chỉnh `BASE_DIR`, `W2V_PATH` và tham số `binary=False` như hướng dẫn trên.
3. Chạy lần lượt các cell khởi tạo, tiền xử lý, xây mô hình.
4. Bỏ comment các dòng `run_experiment(...)` (và/hoặc `run_experiment_v2(...)`) để huấn luyện từng mô hình.
5. Kết quả sẽ hiển thị DataFrame gồm `best_val_acc`, `test_acc`, `test_macro_f1` (ở bản v2) và số tham số. Có thể in thêm `classification_report` nếu cần.

## Ghi chú hữu ích
- Nếu thiếu RAM, giảm `MAX_VOCAB_SIZE`, `MAX_SEQUENCE_LENGTH` hoặc đặt `trainable=False` cho embedding trong bản v2.
- Nếu embedding không load được do định dạng, chuyển file `.txt` sang `.bin` bằng gensim hoặc đổi `binary=False` như trên.
- Có thể tăng dropout/giảm filters khi thấy overfit (`val_loss` tăng, `test_acc` thấp hơn nhiều so với `val_acc`).
- Tất cả dữ liệu/embedding đã nằm trong repo; không cần tải thêm ngoài việc cài thư viện.
