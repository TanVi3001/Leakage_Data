# Leakage_Data

Dự án thực nghiệm **Data Leakage trong bài toán phân loại ảnh bằng PyTorch**, gồm hai notebook độc lập:

- `Mnist_leakageData.ipynb` — demo Data Leakage trên MNIST.
- `Flower_leakageData.ipynb` — demo Data Leakage trên Flower Classification.

Mục tiêu của project là làm rõ vì sao một mô hình có thể đạt kết quả kiểm thử cao hơn thực tế khi dữ liệu Test vô tình xuất hiện trong quá trình huấn luyện, tiền xử lý hoặc lựa chọn mô hình.

---

## 1. Data Leakage là gì?

**Data Leakage** xảy ra khi mô hình sử dụng thông tin mà ở thời điểm dự đoán thực tế nó đáng lẽ không được biết.

Ví dụ đơn giản:

```text
TRAIN ─────► MODEL ─────► TEST
                         ↑
                    đúng quy trình
```

Nếu dữ liệu Test bị đưa ngược vào Train:

```text
TEST ─────► TRAIN ─────► MODEL ─────► TEST
```

thì Test không còn độc lập. Accuracy có thể tăng nhưng không còn phản ánh đúng khả năng **generalization** của mô hình.

---

## 2. Công nghệ sử dụng

- Python
- PyTorch
- TorchVision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab / Jupyter Notebook

---

## 3. Cấu trúc project

```text
Leakage_Data/
│
├── Mnist_leakageData.ipynb
├── Flower_leakageData.ipynb
└── README.md
```

---

# Bài toán 1 — MNIST Data Leakage

## Dataset

MNIST là bộ dữ liệu phân loại chữ số viết tay từ `0` đến `9`.

Thông thường:

```text
Train: 60,000 ảnh
Test : 10,000 ảnh
Classes: 10
```

## Mô hình

MNIST sử dụng một mạng CNN tự xây dựng bằng PyTorch.

Pipeline cơ bản:

```text
Input 1×28×28
      ↓
Conv2D
      ↓
ReLU
      ↓
MaxPool
      ↓
Conv2D
      ↓
ReLU
      ↓
MaxPool
      ↓
Fully Connected
      ↓
10 classes
```

## Thí nghiệm CLEAN

Trong phiên bản CLEAN:

```text
MNIST Train
     ↓
Train / Validation
     ↓
CNN
     ↓
chọn mô hình bằng Validation
     ↓
Test chỉ sử dụng ở cuối
```

Test không xuất hiện trong quá trình training.

## Thí nghiệm LEAKAGE

Trong phiên bản LEAKAGE, một phần dữ liệu Test được cố tình đưa vào Training.

```text
Train
  +
một phần Test
      ↓
     CNN
      ↓
đánh giá lại trên Test
```

Sau đó Test được chia thành:

```text
Seen Test   = ảnh Test đã xuất hiện trong Train
Unseen Test = ảnh Test chưa từng xuất hiện trong Train
```

Mục đích là so sánh:

```text
Accuracy(Seen Test)
        vs
Accuracy(Unseen Test)
```

Nếu mô hình hoạt động tốt hơn rõ rệt trên Seen Test, điều đó cho thấy dữ liệu bị rò rỉ đã làm kết quả đánh giá trở nên lạc quan hơn.

---

# Bài toán 2 — Flower Classification Data Leakage

## Dataset

Dataset sử dụng:

**Flower Classification**  
Kaggle: https://www.kaggle.com/datasets/marquis03/flower-classification

Dataset gồm **14 lớp hoa**.

Dữ liệu được đọc bằng:

```python
torchvision.datasets.ImageFolder
```

---

## Mô hình CNN

Flower Classification không sử dụng pretrained ResNet mà sử dụng CNN tự xây dựng.

Kiến trúc chính:

```text
Input RGB 3×128×128
        ↓
Conv2D 3 → 32
BatchNorm
ReLU
MaxPool
        ↓
Conv2D 32 → 64
BatchNorm
ReLU
MaxPool
        ↓
Conv2D 64 → 128
BatchNorm
ReLU
MaxPool
        ↓
Conv2D 128 → 256
BatchNorm
ReLU
MaxPool
        ↓
Adaptive Average Pool
        ↓
Linear 256 → 128
        ↓
Dropout
        ↓
Linear 128 → 14
```

Loss:

```python
nn.CrossEntropyLoss()
```

Optimizer:

```python
torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
```

---

## Flower CLEAN

Dữ liệu gốc được chia trước thành:

```text
Original Images
      ↓
  Split FIRST
 /     |      \
Train  Val    Test
  ↓
Augmentation
chỉ áp dụng
cho Train
```

Validation được dùng để chọn model tốt nhất.

Test chỉ được sử dụng sau khi training hoàn tất.

Trong thí nghiệm hiện tại, CNN được train trong **5 epochs**.

### Kết quả CLEAN

```text
Flower CLEAN Test Accuracy = 54.07%
```

Đây là baseline vì Test hoàn toàn độc lập với Train.

---

# Flower LEAKAGE

Để demo Data Leakage, project cố tình lấy khoảng **50% ảnh Test** đưa vào Training.

Ví dụ:

```text
Clean:

Train = A B C D E
Test  = X Y Z W
```

Sau khi tạo leakage:

```text
Leakage:

Train = A B C X Y
Test  = X Y Z W
```

Như vậy:

```text
Train ∩ Test ≠ ∅
```

Model đã nhìn thấy một phần ảnh Test trong quá trình training.

Để so sánh công bằng:

- Kiến trúc CNN giữ nguyên.
- Optimizer giữ nguyên.
- Learning rate giữ nguyên.
- Batch size giữ nguyên.
- Số epoch giữ nguyên.
- Validation vẫn giữ sạch.
- Test set cuối vẫn là cùng Test set ban đầu.

Khác biệt chính là **Train của mô hình LEAKAGE chứa một phần ảnh Test**.

---

## Kết quả Flower

| Experiment | Accuracy |
|---|---:|
| Flower CLEAN | **54.07%** |
| Flower LEAKAGE — Full Test | **62.32%** |
| Flower LEAKAGE — Seen Test | **64.21%** |
| Flower LEAKAGE — Unseen Test | **60.43%** |

### Nhận xét

- **Flower CLEAN — 54.07%:** Test không xuất hiện trong Train nên đây là kết quả baseline đáng tin cậy.
- **LEAKAGE Full Test — 62.32%:** tăng khoảng **8.24 điểm phần trăm** so với CLEAN.
- **Seen Test — 64.21%:** cao nhất vì đây là các ảnh Test đã xuất hiện trong Training.
- **Unseen Test — 60.43%:** thấp hơn Seen Test khoảng **3.78 điểm phần trăm**.

Kết quả cho thấy mô hình có lợi thế trên các mẫu đã bị rò rỉ vào Training.

Điểm quan trọng không phải là Data Leakage luôn làm Accuracy tăng một mức cố định, mà là:

> Khi Test đã tham gia vào Training, Test không còn độc lập và metric thu được không còn phản ánh chính xác khả năng generalization trên dữ liệu hoàn toàn mới.

---

# CLEAN vs LEAKAGE

```text
CLEAN

Train ─────────────► CNN
Validation ────────► chọn best model
Test ──────────────► final evaluation

Train ∩ Test = ∅
```

```text
LEAKAGE

             ┌──────── Test samples
             ↓
Train ───────+────────► CNN
                         ↓
Test ─────────────────► Evaluation

Train ∩ Test ≠ ∅
```

---

## Một dạng Leakage thường gặp trong Computer Vision

Ngoài việc trực tiếp đưa ảnh Test vào Train, Data Leakage còn có thể xảy ra khi augmentation được thực hiện **trước khi split**.

Ví dụ:

```text
rose_001.jpg        → Train
flip(rose_001.jpg)  → Test
```

Hai file khác nhau nhưng vẫn có cùng nguồn thông tin.

Cách đúng:

```text
Original Images
      ↓
Split trước
      ↓
Train / Val / Test
      ↓
Augmentation chỉ trên Train
```

---

## Các nguyên tắc hạn chế Data Leakage

1. Split dữ liệu trước khi thực hiện các bước có thể học thông tin từ dữ liệu.
2. Không đưa Test vào Training.
3. Không dùng Test để chọn learning rate, số epoch hoặc kiến trúc mạng.
4. Early stopping chỉ dựa trên Validation.
5. Augmentation chỉ áp dụng sau khi đã split dữ liệu.
6. Kiểm tra duplicate giữa Train / Validation / Test.
7. Với ảnh cùng một đối tượng hoặc cùng phiên chụp, nên split theo group.
8. Chỉ sử dụng Test để đánh giá mô hình cuối cùng.

---

## Kết luận

Hai notebook minh họa Data Leakage ở hai mức độ:

### MNIST

Demo trực tiếp:

```text
Test samples → Training
```

Mục tiêu là hiểu bản chất cơ bản của Train-Test contamination.

### Flower Classification

Demo theo bối cảnh Computer Vision:

```text
Test images → Training
        ↓
Seen Test vs Unseen Test
```

Kết quả Flower cho thấy Test Accuracy có thể tăng đáng kể khi dữ liệu Test bị rò rỉ vào Training.

Vì vậy:

> **Accuracy cao không đồng nghĩa với mô hình tốt nếu quy trình đánh giá bị Data Leakage.**

Một mô hình chỉ được xem là generalize tốt khi được đánh giá trên dữ liệu thật sự độc lập và chưa từng ảnh hưởng đến quá trình xây dựng mô hình.

---

## Dataset

Flower Classification:  
https://www.kaggle.com/datasets/marquis03/flower-classification

MNIST được tải trực tiếp thông qua `torchvision.datasets.MNIST`.

---

## Author

Project thực nghiệm về **Data Leakage trong Image Classification sử dụng PyTorch**.
