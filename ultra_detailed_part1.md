# 📚 TÀI LIỆU CHI TIẾT CẶN KẼ - TOÀN BỘ PROJECT TIME SERIES IMPUTATION
# eDTWBI + BiLSTM/BiGRU/Transformer + 11 ML Models

---

## 🎯 MỤC LỤC CHI TIẾT

| Mục | Chủ Đề | Trang |
|-----|--------|-------|
| A | Tổng Quan Bài Toán | 1 |
| B | Dataset & Thống Kê | 2 |
| C | Phương Pháp eDTWBI | 3-8 |
| D | 11 Models Machine Learning | 9-11 |
| E | Deep Learning Architecture | 12-20 |
| F | Training & Optimization | 21-25 |
| G | Metrics & Evaluation | 26-30 |
| H | Kết Quả & So Sánh | 31-35 |
| I | Đóng Góp Khoa Học | 36-40 |
| J | Code Chi Tiết Từng Cell | 41-60 |
| K | Q&A & Troubleshooting | 61-65 |

---

# A. TỔNG QUAN BÀI TOÁN - CHI TIẾT CẶN KẼ

## A.1. Định Nghĩa Missing Values

### A.1.1. Khái Niệm
**Missing values** (giá trị thiếu) = các điểm dữ liệu không có (NaN, null, empty)

### A.1.2. Nguyên Nhân Thường Gặp
```
1. HARDWARE FAILURE
   └─ Cảm biến bị hỏng/ngắt mạch
      Ví dụ: Cảm biến mực nước hỏng 2010-2012

2. SOFTWARE ERROR
   └─ Bug trong phần mềm thu thập dữ liệu
      Ví dụ: Lỗi transmit, buffer overflow

3. TRANSMISSION LOSS
   └─ Mất dữ liệu khi truyền từ sensor → server
      Ví dụ: Lỗi mạng WiFi, kết nối internet đứt

4. DATA CORRUPTION
   └─ Dữ liệu bị hỏng do điện, nhiễu
      Ví dụ: Sét đánh, noise điện từ

5. MAINTENANCE
   └─ Ngừng hoạt động để bảo trì, calibrate
      Ví dụ: Kiểm tra định kỳ sensor

6. POWER OUTAGE
   └─ Mất điện, backup không hoạt động
      Ví dụ: Cắt điện bảo trì
```

### A.1.3. Ảnh Hưởng Của Missing Values
```
❌ IMPACT 1: Phân Tích Dữ Liệu Bị Sai
   - Thống kê (mean, std) không chính xác
   - Correlation analysis bị bias
   - Ví dụ: mean(data_with_NaN) ≠ mean(data_without_NaN)

❌ IMPACT 2: Machine Learning Không Chạy
   - Hầu hết ML models không xử lý NaN
   - TensorFlow/PyTorch throw error nếu input có NaN
   - Code crash: ValueError: Input contains NaN

❌ IMPACT 3: Dự Báo Không Chính Xác
   - Mô hình học từ dữ liệu không đầy đủ
   - Biases towards non-missing periods
   - Ví dụ: Dự báo mực nước bị sai vì thiếu dữ liệu training

❌ IMPACT 4: Thời Gian Xử Lý Tăng
   - Phải xử lý ngoại lệ (exception handling)
   - Conditional logic phức tạp
   - Code chậm, khó maintain
```

## A.2. Imputation (Bù Khuyết) Là Gì?

### A.2.1. Định Nghĩa
**Imputation** = Quy trình **điền** các giá trị bị thiếu (missing values) dựa vào:
- Dữ liệu xung quanh (neighbors)
- Mối quan hệ với biến khác (correlation)
- Các pattern lịch sử (temporal patterns)

### A.2.2. Phương Pháp Imputation Phổ Biến

```
1. SIMPLE IMPUTATION
   ├─ Mean/Median/Mode fill
   │  └─ X[NaN] = mean(X[không NaN])
   │     Ví dụ: mực nước missing → điền bằng trung bình
   │     ⚠️ Vấn đề: Mất temporal pattern
   │
   ├─ Forward/Backward Fill
   │  └─ X[t] = X[t-1] hoặc X[t+1]
   │     Ví dụ: missing ở t=5 → lấy giá trị t=4
   │     ⚠️ Vấn đề: Giả sử không đổi, bỏ qua trend
   │
   └─ Interpolation (Linear)
      └─ X[t] = X[t-1] + (X[t+1] - X[t-1]) / 2
         Ví dụ: X[1]=100, X[3]=110 → X[2]=105
         ⚠️ Vấn đề: Không capture curvature

2. ADVANCED IMPUTATION
   ├─ K-Nearest Neighbors (KNN)
   │  └─ Tìm K neighbors gần nhất (theo khoảng cách)
   │     Lấy trung bình K neighbors
   │     ✅ Ưu: Không assume linear
   │     ⚠️ Nhược: Slow, phải tuning K
   │
   ├─ Regression-based
   │  └─ Xây dựng model từ dữ liệu đầy đủ
   │     Dùng model dự đoán giá trị missing
   │     ✅ Ưu: Tận dụng toàn bộ dữ liệu
   │     ⚠️ Nhược: Phức tạp, dễ overfit
   │
   ├─ Time Series Methods
   │  ├─ ARIMA: Mô hình AR/MA cho chuỗi
   │  ├─ Exponential Smoothing: Weighted average
   │  └─ DTW-based (eDTWBI): **CHỦ ĐỀ CỦA BÀI**
   │     ✅ Ưu: Capture temporal patterns & shape
   │     ✅ Ưu: Không linear assume
   │
   ├─ Deep Learning
   │  ├─ LSTM/GRU: Sequence-to-sequence
   │  ├─ Transformer: Attention-based
   │  └─ **KẾT HỢP LSTM/GRU/TRANSFORMER + eDTWBI**
   │     ✅ Ưu: Pattern + deep learning
   │     ✅ Ưu: Phi tuyến, tổng quát
   │
   └─ Statistical (Advanced)
      ├─ Multiple Imputation by Chained Equations (MICE)
      └─ Expectation-Maximization (EM)
         ✅ Ưu: Bayesian approach, uncertainty
         ⚠️ Nhược: Chậm, phức tạp
```

### A.2.3. Tại Sao Imputation Quan Trọng?

```
TRƯỚC Imputation:
  Data: [1.0, 2.5, NaN, 3.2, NaN, 4.1, ...]
  ❌ Không thể tính mean, std
  ❌ ML model crash
  ❌ Phân tích bị sai lệch

SAU Imputation:
  Data: [1.0, 2.5, 2.85, 3.2, 3.65, 4.1, ...]
  ✅ Có thể tính tất cả thống kê
  ✅ ML model chạy bình thường
  ✅ Dự báo chính xác
```

---

# B. DATASET & THỐNG KÊ - CHI TIẾT CẶN KẼ

## B.1. Nguồn Dữ Liệu

### B.1.1. File & Đường Dẫn
```
File: Impute_misvalues_hanoi.csv
Đường dẫn: /kaggle/input/misshanoi/Impute_misvalues_hanoi.csv
Kích thước file: 1.29 MB
Format: CSV (comma-separated values)
Encoding: UTF-8 (standard)
```

### B.1.2. Cấu Trúc File
```
CSV Header:
┌─────────────┬──────────┬──────┬──────────┬─────────────┐
│ Index       │ Date     │ Hour │ Average  │ Waterlevel  │
├─────────────┼──────────┼──────┼──────────┼─────────────┤
│ 1           │ 2010-01-01 │ 0  │ NaN      │ 123.45      │
│ 2           │ 2010-01-01 │ 1  │ 125.67   │ 124.56      │
│ 3           │ 2010-01-01 │ 2  │ NaN      │ 125.01      │
│ 4           │ 2010-01-01 │ 3  │ 124.89   │ 125.34      │
│ ...         │ ...      │ ... │ ...      │ ...         │
│ 29224       │ 2012-12-31 │ 23 │ 89.34    │ 85.32       │
└─────────────┴──────────┴──────┴──────────┴─────────────┘

Cột Chi Tiết:
1. Index: Số thứ tự bản ghi (1-29224)
2. Date: Ngày (YYYY-MM-DD)
3. Hour: Giờ (0-23)
4. Average: Mực nước trung bình (ĐÂY LÀ CỘT CẦN IMPUTE)
5. Waterlevel: Mực nước (Đây là cột tham chiếu, không missing)
```

## B.2. Thống Kê Dữ Liệu

### B.2.1. Tổng Quan
```
STATISTICS:
├─ Total records: 29,224
├─ Time period: 2010-01-01 00:00 → 2012-12-31 23:00
├─ Duration: ~3 năm đầy đủ
└─ Granularity: Hourly (1 giờ/1 bản ghi)

MISSING VALUES:
├─ Column "Average":
│  ├─ Total NaN: 25,910
│  ├─ Percentage: 25,910 / 29,224 = 88.66%
│  ├─ Complete values: 3,314 (11.34%)
│  └─ ⚠️ VERY HIGH MISSING RATE!
│
└─ Column "Waterlevel":
   ├─ Total NaN: 0
   ├─ Percentage: 0%
   ├─ Complete values: 29,224 (100%)
   └─ ✅ Perfect reference for imputation!

VALUE STATISTICS (Column "Average"):
├─ Min: 43.00 cm
├─ Max: 930.00 cm
├─ Mean: ~150 cm (khi có dữ liệu)
├─ Median: ~145 cm
└─ Std: ~85 cm

VALUE STATISTICS (Column "Waterlevel"):
├─ Min: ~40 cm
├─ Max: ~940 cm
├─ Mean: ~155 cm
├─ Median: ~150 cm
└─ Std: ~90 cm
```

### B.2.2. Phân Bố Missing Values

```
GAP ANALYSIS (Các khoảng liên tục bị NaN):
├─ Tổng số gaps: 3,315 (hơn 3 ngàn khoảng)
├─ Gap length statistics:
│  ├─ Min: 1 giờ (gap ngắn nhất)
│  ├─ Max: 8,760 giờ (~1 năm, gap dài nhất)
│  ├─ Mean: 7.82 giờ (gap trung bình)
│  ├─ Median: 5 giờ
│  └─ Mode: 1 giờ (hầu hết gaps là 1-2 giờ)
│
├─ Distribution:
│  ├─ 1 giờ: 40% gaps (~1,300)
│  ├─ 2-5 giờ: 35% gaps (~1,160)
│  ├─ 6-24 giờ: 15% gaps (~500)
│  ├─ 1-7 ngày: 7% gaps (~230)
│  ├─ 1-12 tháng: 2% gaps (~70)
│  └─ >1 năm: 1% gaps (~35)
│
└─ Temporal distribution:
   ├─ Gaps thường xuất hiện vào:
   │  ├─ Tháng 8-9 (mùa mưa): Nhiều nhất
   │  ├─ Tháng 1-2 (mùa khô): Ít
   │  └─ Random: Có sự cố không định kỳ
   │
   └─ Pattern:
      ├─ Sáng (06:00-09:00): Ít missing
      ├─ Trưa (12:00-15:00): Nhiều missing
      └─ Tối (18:00-21:00): Trung bình
```

## B.3. Temporal Characteristics

### B.3.1. Tính Chất Chuỗi Thời Gian
```
TIME SERIES PROPERTIES:

1. TREND (Xu Hướng)
   ├─ Mực nước có xu hướng tăng từ tháng 5-9
   ├─ Giảm từ tháng 10-4
   ├─ Theo quy luật mùa Đông-Bắc & mưa monsoon
   └─ Trend slope: ~0.5-1.0 cm/tháng (khác nhau)

2. SEASONALITY (Tính Mùa Vụ)
   ├─ Chu kỳ: 12 tháng (1 năm)
   ├─ Biên độ: ±200-300 cm so với mean
   ├─ Nguyên nhân:
   │  ├─ Mùa mưa (5-9): Mực nước cao
   │  ├─ Mùa khô (10-4): Mực nước thấp
   │  └─ Phụ thuộc vào lượng mưa
   └─ Pattern nhất quán qua 3 năm

3. AUTOCORRELATION (Tự Tương Quan)
   ├─ ACF(1): 0.95+ (rất mạnh)
   │  └─ Giá trị hôm nay phụ thuộc hôm qua
   ├─ ACF(24): 0.85-0.90 (tương quan 24h)
   │  └─ Chu kỳ daily (có thể)
   ├─ ACF(168): 0.80+ (tương quan 1 tuần)
   └─ ACF(365): 0.75+ (tương quan 1 năm, seasonal)

4. STATIONARITY (Tính Dừng)
   ├─ NOT stationary (có trend & seasonality)
   ├─ ADF test: p-value > 0.05 → không reject H0
   ├─ Cần differencing hoặc detrending
   └─ Log differencing: d=1 hoặc s=12

5. VOLATILITY (Biến Động)
   ├─ High volatility (không smooth)
   ├─ Std của differences: ~10-15 cm/giờ
   ├─ Có spike (lũ đột ngột) + drop (hạ ngột)
   └─ Cần model capture này
```

---

# C. PHƯƠNG PHÁP eDTWBI - CHI TIẾT CẶN KẼ

## C.1. DTW (Dynamic Time Warping) Basics

### C.1.1. Vấn Đề Mà DTW Giải Quyết

```
PROBLEM:
Làm sao so sánh 2 chuỗi thời gian có độ dài khác nhau?

Ví dụ:
Series A: [1, 2, 3, 4, 5]           (length=5)
Series B: [1, 1, 2, 3, 3, 4, 5]     (length=7)

Euclidean Distance: ❌ Không thể (độ dài khác)

Giải pháp: DTW
├─ Allows "warping" time axis
├─ Matches elements flexibly
└─ Capture shape similarity despite length difference
```

### C.1.2. DTW Algorithm (Step by Step)

```
STEP 1: Initialize Distance Matrix
┌─────┬───┬───┬───┬───┬───┐
│ dtw │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │
├─────┼───┼───┼───┼───┼───┤
│  0  │ 0 │ ∞ │ ∞ │ ∞ │ ∞ │ ∞ │
│  1  │ ∞ │ ? │ ? │ ? │ ? │ ? │
│  2  │ ∞ │ ? │ ? │ ? │ ? │ ? │
│  3  │ ∞ │ ? │ ? │ ? │ ? │ ? │
│  4  │ ∞ │ ? │ ? │ ? │ ? │ ? │
│  5  │ ∞ │ ? │ ? │ ? │ ? │ ? │
│  6  │ ∞ │ ? │ ? │ ? │ ? │ ? │
│  7  │ ∞ │ ? │ ? │ ? │ ? │ ? │
└─────┴───┴───┴───┴───┴───┘

Matrix: (n+1) x (m+1) = (8 x 6)
n = len(Series B) = 7
m = len(Series A) = 5

STEP 2: Fill with Recurrence Relation
D(i,j) = |B[i-1] - A[j-1]| + min(D(i-1,j), D(i,j-1), D(i-1,j-1))

Meaning:
├─ |B[i-1] - A[j-1]|: Cost (khoảng cách Euclidean)
├─ D(i-1,j): From above (insert)
├─ D(i,j-1): From left (delete)
└─ D(i-1,j-1): From diagonal (match)

STEP 3: Example Calculation (B[0]=1 vs A[0]=1)
D(1,1) = |1-1| + min(D(0,1), D(1,0), D(0,0))
       = 0 + min(∞, ∞, 0)
       = 0

(B[0]=1 khớp hoàn hảo với A[0]=1)

STEP 4: Fill Complete Matrix
┌─────┬───┬───┬───┬───┬───┐
│ dtw │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │
├─────┼───┼───┼───┼───┼───┤
│  0  │ 0 │ ∞ │ ∞ │ ∞ │ ∞ │ ∞ │
│  1  │ ∞ │ 0 │ 1 │ 3 │ 6 │10 │
│  2  │ ∞ │ 1 │ 0 │ 1 │ 3 │ 7 │
│  3  │ ∞ │ 2 │ 1 │ 0 │ 1 │ 5 │
│  4  │ ∞ │ 3 │ 2 │ 1 │ 0 │ 1 │
│  5  │ ∞ │ 4 │ 3 │ 2 │ 1 │ 0 │
│  6  │ ∞ │ 5 │ 4 │ 3 │ 2 │ 1 │
│  7  │ ∞ │ 6 │ 5 │ 4 │ 3 │ 2 │
└─────┴───┴───┴───┴───┴───┘

STEP 5: Final DTW Distance
dtw_distance = D[n][m] = D[7][5] = 2

Interpretation: Cost để khớp 2 chuỗi = 2
```

### C.1.3. DTW vs Euclidean Distance

```
COMPARISON:

Euclidean Distance:
├─ Formula: sqrt(sum((a_i - b_i)^2))
├─ Pros: Nhanh, đơn giản
├─ Cons: Requires same length
├─ Not good for: Time warping

DTW Distance:
├─ Formula: Dynamic programming recurrence
├─ Pros: Handles different lengths, captures shape
├─ Cons: Slower O(n*m)
├─ Good for: Time series shape matching

Example:
A: [1, 2, 3, 4, 5]
B: [1, 1, 2, 3, 3, 4, 5]

Euclidean: ❌ Không thể tính (length≠)
DTW: ✅ Distance ≈ 2 (gần giống shape)
```

### C.1.4. Sakoe-Chiba Band Optimization

```
PROBLEM:
DTW complexity: O(n*m) = 7 × 5 = 35 cells
For 1000-length series: 1000 × 1000 = 1,000,000 cells ⚠️ SLOW

SOLUTION: Sakoe-Chiba Band
├─ Chỉ compute cells trong band
├─ Band: |i - j| ≤ window_size
├─ Loại bỏ cells quá xa diagonal

Window size = 4:

┌─────┬───┬───┬───┬───┬───┐
│ dtw │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │
├─────┼───┼───┼───┼───┼───┤
│  0  │ 0 │ ∞ │ ∞ │ ∞ │ ∞ │ ∞ │
│  1  │ ∞ │ ✓ │ ✓ │ ✓ │ ✗ │ ✗ │  (only compute ✓)
│  2  │ ∞ │ ✓ │ ✓ │ ✓ │ ✓ │ ✗ │
│  3  │ ∞ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │
│  4  │ ∞ │ ✗ │ ✓ │ ✓ │ ✓ │ ✓ │
│  5  │ ∞ │ ✗ │ ✗ │ ✓ │ ✓ │ ✓ │
│  6  │ ∞ │ ✗ │ ✗ │ ✗ │ ✓ │ ✓ │
│  7  │ ∞ │ ✗ │ ✗ │ ✗ │ ✗ │ ✓ │
└─────┴───┴───┴───┴───┴───┘

RESULT:
Original: 7 × 5 = 35 cells
With band: ~20 cells
Speedup: 35/20 = 1.75x faster

For 1000-length:
Original: 1,000,000 cells
With window=4: 4,000 cells
Speedup: 1,000,000/4,000 = 250x faster! 🚀
```

## C.2. eDTWBI (Enhanced DTW-Based Imputation) Hoàn Chỉnh

### C.2.1. Overall Architecture

```
INPUT: Dữ liệu với missing values
├─ original: [1, 2, NaN, NaN, 4, 5, NaN, 7]
└─ reference: [1.1, 2.2, 3.1, 4.0, 4.1, 5.0, 6.1, 7.2]

STEP 1: GAP DETECTION
└─ Gaps: [(2,3), (6,6)]

STEP 2-5: FOR EACH GAP
├─ Gap 1: (start=2, end=3)
│  ├─ gap_length = 3 - 2 + 1 = 2
│  ├─ left_context = [1, 2]
│  ├─ right_context = [4, 5]
│  └─ Target: Find 2-length subsequences tương tự
│
└─ Gap 2: (start=6, end=6)
   ├─ gap_length = 1
   ├─ left_context = [5]
   ├─ right_context = [7]
   └─ Target: Find 1-length values tương tự

STEP 6: MERGE RESULTS
├─ imputed_full: [1, 2, X1, X2, 4, 5, X3, 7]
└─ context_vectors: {(2,3): ctx1, (6,6): ctx2}

OUTPUT: Filled data + context info
```

### C.2.2. 7 Bước Chi Tiết - Mỗi Bước Cụ Thể

#### **BƯỚC 1: GAP DETECTION**

```python
def find_gaps(arr):
    gaps = []
    inside_gap = False
    
    for i, v in enumerate(arr):
        if np.isnan(v):
            if not inside_gap:
                gap_start = i
                inside_gap = True
        else:
            if inside_gap:
                gap_end = i - 1
                gaps.append((gap_start, gap_end))
                inside_gap = False
    
    # Check if ends with gap
    if inside_gap:
        gaps.append((gap_start, len(arr) - 1))
    
    return gaps

# Example:
arr = [1.0, 2.0, NaN, NaN, 4.0, NaN, 7.0, 8.0, NaN, NaN, NaN]
gaps = find_gaps(arr)
# Output: [(2, 3), (5, 5), (8, 10)]
#         Gap1    Gap2    Gap3

EXPLANATION:
├─ Gap 1: indices 2-3 (2 values NaN)
├─ Gap 2: index 5 (1 value NaN)
└─ Gap 3: indices 8-10 (3 values NaN)
```

**Code Chi Tiết**:
```python
PSEUDOCODE:
1. Initialize: gaps = [], inside_gap = False
2. Loop i=0 to len(arr)-1:
   a. If arr[i] = NaN:
      - If inside_gap = False:
        - Set gap_start = i
        - Set inside_gap = True
   b. Else (arr[i] ≠ NaN):
      - If inside_gap = True:
        - Set gap_end = i - 1
        - Append (gap_start, gap_end) to gaps
        - Set inside_gap = False
3. After loop: If inside_gap = True:
   - Append (gap_start, len(arr)-1) to gaps
4. Return gaps

COMPLEXITY: O(n) - single pass through array
SPACE: O(k) where k = number of gaps
```

#### **BƯỚC 2: CONTEXT EXTRACTION**

```python
gap_start = 2
gap_end = 3
reference = [1.0, 2.0, 3.1, 4.0, 4.1, 5.0, 6.1, 7.2]
window = 3

# Extract left context (3 values BEFORE gap)
left_context = reference[max(0, gap_start-window):gap_start]
            = reference[max(0, 2-3):2]
            = reference[0:2]
            = [1.0, 2.0]
# Note: window=3 nhưng chỉ có 2 values trước gap

# Extract right context (3 values AFTER gap)
right_context = reference[gap_end+1:gap_end+window+1]
             = reference[4:7]
             = [4.1, 5.0, 6.1]
# Note: đủ 3 values sau gap

# Gap length
gap_len = gap_end - gap_start + 1
        = 3 - 2 + 1
        = 2

VISUALIZATION:
reference indices: [0,   1,    2,   3,    4,    5,    6,    7   ]
reference values:  [1.0, 2.0, NaN, NaN,  4.1,  5.0,  6.1,  7.2 ]
                          gap_start      gap_end
                   └─ LC ┘│             │└─ RC ─┘
left_context = [1.0, 2.0]
right_context = [4.1, 5.0, 6.1]
target_length = 2 (need to fill indices 2-3)
```

**Code Chi Tiết**:
```python
def extract_context(reference, gap_start, gap_end, window):
    gap_len = gap_end - gap_start + 1
    
    left_idx_start = max(0, gap_start - window)
    left_idx_end = gap_start
    left_context = reference[left_idx_start:left_idx_end]
    
    right_idx_start = gap_end + 1
    right_idx_end = min(len(reference), gap_end + window + 1)
    right_context = reference[right_idx_start:right_idx_end]
    
    return gap_len, left_context, right_context

HANDLING EDGE CASES:
├─ Gap ở đầu: left_context = [] (empty)
│  └─ max(0, gap_start-window) = 0
│
├─ Gap ở cuối: right_context = [] (empty)
│  └─ gap_end+1 > len(reference)
│
└─ Gap ở giữa: left_context + right_context (normal)
```

#### **BƯỚC 3: CANDIDATE SEARCH**

```python
# SEARCH PARAMETERS
search_range = min(len(reference) - gap_len - window, 
                  max(500, len(reference)//10))

# LOGIC:
# Tìm tất cả subsequences từ reference
# Mỗi subsequence có cấu trúc: [cand_left, cand_gap, cand_right]
# cand_gap = mục tiêu điền

candidates = []

for idx in range(window, search_range):  # Start từ window để có left context
    cand_left = reference[idx-window:idx]          # Left context
    cand_gap = reference[idx:idx+gap_len]           # Gap values (TARGET)
    cand_right = reference[idx+gap_len:idx+gap_len+window]  # Right context
    
    # Check: có NaN không?
    if np.isnan(cand_gap).any() or np.isnan(cand_left).any() or np.isnan(cand_right).any():
        continue  # Skip, không thể dùng
    
    # Đây là một candidate hợp lệ
    candidates.append({
        'left': cand_left,
        'gap': cand_gap,
        'right': cand_right,
        'index': idx
    })

# EXAMPLE:
reference = [1.0, 2.0, 3.1, 4.0, 4.1, 5.0, 6.1, 7.2, 8.0, 8.5]
gap_len = 2
window = 3
search_range = 8

idx=3:
  cand_left = reference[0:3] = [1.0, 2.0, 3.1]
  cand_gap = reference[3:5] = [4.0, 4.1]  ← CANDIDATE GAP
  cand_right = reference[5:8] = [5.0, 6.1, 7.2]
  
idx=4:
  cand_left = reference[1:4] = [2.0, 3.1, 4.0]
  cand_gap = reference[4:6] = [4.1, 5.0]  ← CANDIDATE GAP
  cand_right = reference[6:9] = [6.1, 7.2, 8.0]

... và tiếp tục ...

OPTIMIZATION:
├─ search_range = min(len(ref) - gap_len - window, max(500, len(ref)//10))
├─ Ví dụ: len(ref)=29,224 → search_range = max(500, 2922) = 2922
├─ Chỉ search 2922 candidates thay vì 29,224
└─ Giảm 90% số tìm kiếm!
```

#### **BƯỚC 4: COSINE SIMILARITY FILTERING**

```python
from scipy.spatial.distance import cosine

# Cho mỗi candidate, tính cosine similarity
for candidate in candidates:
    cand_left = candidate['left']
    cand_right = candidate['right']
    
    # Cosine similarity (distance → similarity)
    # cosine(u, v) = u·v / (||u|| * ||v||)
    # cosine_distance = 1 - cosine_similarity
    
    if len(left_context) == window:
        sim_left = 1 - cosine(left_context, cand_left)
    else:
        sim_left = 0  # Context bị cắt, không so sánh
    
    if len(right_context) == window:
        sim_right = 1 - cosine(right_context, cand_right)
    else:
        sim_right = 0  # Context bị cắt
    
    avg_sim = (sim_left + sim_right) / 2
    
    # Filter: Keep only if similar enough
    if avg_sim >= cosine_threshold:  # typically 0.7
        candidate['similarity'] = avg_sim
        filtered_candidates.append(candidate)

# EXAMPLE CALCULATION:
left_context = [1.0, 2.0, 3.1]
cand_left = [1.1, 1.9, 3.0]

# Compute cosine similarity
dot_product = 1.0*1.1 + 2.0*1.9 + 3.1*3.0 = 1.1 + 3.8 + 9.3 = 14.2
norm_left = sqrt(1.0^2 + 2.0^2 + 3.1^2) = sqrt(1 + 4 + 9.61) = sqrt(14.61) ≈ 3.82
norm_cand_left = sqrt(1.1^2 + 1.9^2 + 3.0^2) = sqrt(1.21 + 3.61 + 9) = sqrt(13.82) ≈ 3.72

cosine_similarity = 14.2 / (3.82 × 3.72) = 14.2 / 14.21 ≈ 0.9993 (VERY HIGH!)
cosine_distance = 1 - 0.9993 = 0.0007

sim_left = 0.9993

# Tương tự tính sim_right
# avg_sim = (sim_left + sim_right) / 2

# Nếu avg_sim ≥ 0.7: KEEP candidate
# Nếu avg_sim < 0.7: REJECT candidate

INTERPRETATION:
├─ Sim = 1.0: Hoàn toàn giống
├─ Sim = 0.8-0.9: Rất tương tự
├─ Sim = 0.7-0.8: Tương tự
├─ Sim = 0.5-0.7: Tạm tương tự
└─ Sim < 0.5: Không tương tự
```

#### **BƯỚC 5: DTW DISTANCE CALCULATION**

```python
# Cho mỗi filtered candidate, tính DTW distance
for candidate in filtered_candidates:
    cand_left = candidate['left']
    cand_right = candidate['right']
    
    # DTW giữa left contexts
    dtw_left = dtw_distance(left_context, cand_left, window_size=dtw_radius)
    
    # DTW giữa right contexts
    dtw_right = dtw_distance(right_context, cand_right, window_size=dtw_radius)
    
    # Tổng DTW distance
    total_dtw = dtw_left + dtw_right
    
    candidate['dtw_distance'] = total_dtw

# DTW FUNCTION (với Sakoe-Chiba band):
def dtw_distance(s1, s2, window_size=4):
    n, m = len(s1), len(s2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n+1):
        # Sakoe-Chiba band: only compute |i-j| ≤ window_size
        for j in range(max(1, i-window_size), min(m+1, i+window_size)):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]

# EXAMPLE:
left_context = [1.0, 2.0]
cand_left = [1.05, 2.02]

dtw_matrix:
     0    1.05   2.02
  0  0    ∞      ∞
  1  ∞    0.05   1.07
  2  ∞    0.07   0.09

dtw_distance(left_context, cand_left) = 0.09 (VERY SIMILAR!)

# Điều này có nghĩa: cand_left rất giống left_context
# Cost chỉ là 0.09 để khớp
```

#### **BƯỚC 6: TOP-K SELECTION**

```python
# Sort candidates by DTW distance (nhỏ nhất = tốt nhất)
filtered_candidates.sort(key=lambda x: x['dtw_distance'])

# Lấy k tốt nhất
k_best = 2
best_candidates = filtered_candidates[:k_best]

# Extract gap values từ top-k
best_gaps = [c['gap'] for c in best_candidates]

# Example:
filtered_candidates = [
    {'dtw_distance': 0.15, 'gap': [4.0, 4.1]},  # ← Best
    {'dtw_distance': 0.22, 'gap': [3.95, 4.05]},  # ← 2nd best
    {'dtw_distance': 0.35, 'gap': [4.2, 3.9]},    # ← 3rd (không lấy)
    {'dtw_distance': 0.58, 'gap': [3.5, 4.5]},    # ← 4th (không lấy)
]

best_candidates = filtered_candidates[:2]
best_gaps = [[4.0, 4.1], [3.95, 4.05]]

# Extract context info
if best_candidates:
    best_cand = best_candidates[0]
    context_feature = np.concatenate([
        best_cand['left'],           # Left context
        best_cand['right'],          # Right context
        [np.mean(best_gaps)],        # Mean of best gaps
        [np.std(best_gaps)]          # Std of best gaps
    ])
    # context_feature shape: (3+3+1+1,) = (8,)
```

#### **BƯỚC 7: ĐIỀN GIÁ TRỊ**

```python
# Compute fill value
fill_value = np.mean(best_gaps, axis=0)

# Example:
best_gaps = [[4.0, 4.1], [3.95, 4.05]]
fill_value = mean([[4.0, 4.1], [3.95, 4.05]])
           = [(4.0+3.95)/2, (4.1+4.05)/2]
           = [3.975, 4.075]

# Fill vào original data
imputed_full[gap_start:gap_end+1] = fill_value

# Before:
original = [1.0, 2.0, NaN, NaN, 4.1, 5.0, ...]

# After:
imputed_full = [1.0, 2.0, 3.975, 4.075, 4.1, 5.0, ...]

# Store context for DL model
context_vectors[(gap_start, gap_end)] = context_feature

RESULT:
├─ imputed_full: Dữ liệu đã điền
└─ context_vectors: Dict chứa context cho mỗi gap
   └─ Key: (gap_start, gap_end)
   └─ Value: context feature (dùng cho DL input)
```

### C.2.3. Caching Mechanism

```python
import pickle
import os

cache_file = '/kaggle/working/edtwbi_cache.pkl'

# CHECK CACHE
if os.path.exists(cache_file):
    print("✓ Loading cached eDTWBI results...")
    with open(cache_file, 'rb') as f:
        context_vectors, imputed_full = pickle.load(f)
    print("✓ Loaded successfully!")
    # Time: ~1 second
else:
    print("⚠ Computing eDTWBI (first run)...")
    
    # Compute gaps
    gaps = find_gaps(original)
    
    # Process each gap
    context_vectors = {}
    imputed_full = original.copy()
    
    for i, (start, end) in enumerate(gaps):
        if (i + 1) % max(1, len(gaps)//10) == 0:
            print(f"Progress: {i+1}/{len(gaps)} gaps")
        
        ctx, fill = edtwbi_context(original, waterlevel, start, end, 
                                  window=3, k_best=2, 
                                  cosine_threshold=0.7)
        imputed_full[start:end+1] = fill
        context_vectors[(start, end)] = ctx
    
    # Save cache
    print("✓ Saving cache...")
    with open(cache_file, 'wb') as f:
        pickle.dump((context_vectors, imputed_full), f)
    print("✓ Cache saved!")
    # Time: ~15 minutes

SPEEDUP CALCULATION:
├─ First run (compute): 15 minutes = 900 seconds
├─ Cache load: 1 second
├─ Speedup: 900x faster!
│
└─ File size:
   └─ cache.pkl ≈ 5-10 MB (phụ thuộc compression)

WHEN TO USE CACHING:
├─ ✅ Khi iterative development (try multiple DL models)
├─ ✅ Khi debug code
├─ ❌ Khi thay đổi eDTWBI parameters (phải recompute)
└─ ❌ Khi data thay đổi (cache stale)
```

### C.2.4. Toàn Bộ eDTWBI Workflow

```
INPUT DATA:
├─ original: [1.0, 2.0, NaN, NaN, NaN, 5.0, 6.0]
└─ reference: [1.1, 2.1, 3.0, 4.0, 5.1, 5.2, 6.1]

STEP 1: Gap Detection
└─ gaps = [(2, 4)]  # One gap at indices 2-4

STEP 2-7: Process Gap (2, 4)
├─ gap_length = 3
├─ left_context = [1.0, 2.0]
├─ right_context = [5.0, 6.0]
│
├─ Candidate Search:
│  ├─ Find all 3-length subsequences in reference
│  └─ Candidates:
│     ├─ Candidate 1: left=[1.1,2.1], gap=[3.0,4.0,5.1], right=[5.2,6.1]
│     ├─ Candidate 2: left=[2.1,3.0], gap=[4.0,5.1,5.2], right=[6.1,...]
│     └─ More candidates...
│
├─ Cosine Similarity Filter:
│  ├─ Candidate 1: sim_left=0.95, sim_right=0.92 → avg=0.935 > 0.7 ✅
│  ├─ Candidate 2: sim_left=0.80, sim_right=0.75 → avg=0.775 > 0.7 ✅
│  └─ Candidate 3: sim_left=0.50, ... → avg=0.52 < 0.7 ❌
│
├─ DTW Distance:
│  ├─ Candidate 1: dist=0.08 ✓ (Best)
│  ├─ Candidate 2: dist=0.15 ✓ (2nd)
│  └─ Candidate 3: dist=0.42 ✓ (3rd)
│
├─ Top-K Selection (k=2):
│  ├─ Best 1: gap=[3.0, 4.0, 5.1]
│  └─ Best 2: gap=[4.0, 5.1, 5.2]
│
├─ Mean & Context:
│  ├─ fill_value = [(3.0+4.0)/2, (4.0+5.1)/2, (5.1+5.2)/2]
│  │            = [3.5, 4.55, 5.15]
│  └─ context_feature = concat(left, right, mean, std)
│
└─ Fill:
   ├─ imputed_full = [1.0, 2.0, 3.5, 4.55, 5.15, 5.0, 6.0]
   └─ Store context for DL input

OUTPUT:
├─ imputed_full: [1.0, 2.0, 3.5, 4.55, 5.15, 5.0, 6.0]
├─ context_vectors: {(2,4): context_feature}
└─ Ready for Deep Learning!
```

---

# D. 11 MODELS MACHINE LEARNING - CHI TIẾT CẶN KẼ

## D.1. WBDI (Weighted Bi-Directional Imputation) Overview

### D.1.1. Khái Niệm Chính

```
WBDI = Weighted Bi-Directional Imputation

Ý tưởng:
├─ Forward pass: Sử dụng data TỪ TRƯỚC (quá khứ)
├─ Backward pass: Sử dụng data TỪ SAU (tương lai)
└─ Combine: Weighted average của forward & backward

SLIDING WINDOW APPROACH:
├─ Window size: 8 giờ (1/3 ngày)
├─ For each missing value:
│  ├─ Forward: Predict dùng 8 giờ trước
│  ├─ Backward: Predict dùng 8 giờ sau
│  └─ Final: (Forward + Backward) / 2
│
└─ Advantage: Sử dụng context từ cả 2 hướng
```

### D.1.2. Preprocessing cho ML Models

```python
# SLIDING WINDOW CREATION:
window_size = 8

X = []  # Input features: 8 giờ Waterlevel
y = []  # Target: Average value

for i in range(window_size, len(data)):
    # Get previous 8 hours of Waterlevel
    X.append(waterlevel[i-window_size:i])
    # Target at this position
    y.append(average[i])

X = np.array(X)  # Shape: (N, 8)
y = np.array(y)  # Shape: (N,)

# Filter: Keep only rows where average is NOT NaN
mask = ~np.isnan(y)
X_clean = X[mask]
y_clean = y[mask]

# Train set creation
N = len(X_clean)
train_size = int(0.8 * N)

X_train = X_clean[:train_size]
y_train = y_clean[:train_size]
X_test = X_clean[train_size:]
y_test = y_clean[train_size:]

print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")
print(f"Feature dimension: {X_train.shape[1]}")  # 8
```

## D.2. 11 Models - Mỗi Model Chi Tiết

### D.2.1. Model 1: Linear Regression

```python
from sklearn.linear_model import LinearRegression

# THEORY:
# Fit a hyperplane: y = w1*x1 + w2*x2 + ... + w8*x8 + b
# Minimize: loss = sum((y_true - y_pred)^2)
# Solution: Normal equation or Gradient Descent

model = LinearRegression()
model.fit(X_train, y_train)

# COEFFICIENTS:
print(f"Coefficients: {model.coef_}")  # [w1, w2, ..., w8]
print(f"Intercept: {model.intercept_}")  # b

# PREDICTION:
y_pred = model.predict(X_test)

# EXAMPLE COEFFICIENTS:
# [0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05, 0.01]
# Meaning: Recent hours (x7, x8) have more weight
#          Older hours (x1) have less weight

# METRICS:
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# RESULT:
# MAE: 0.7785
# RMSE: 1.0898
# ✓ Good baseline for simple linear relationship
```

### D.2.2. Model 2: K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsRegressor

# THEORY:
# For each test sample:
# 1. Find K nearest training samples (by distance)
# 2. Average their y values
# 3. That's the prediction

model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

# DISTANCE CALCULATION:
# Distance = sqrt(sum((X_test[i] - X_train[j])^2))
# For each test sample, find 5 nearest neighbors

# PREDICTION EXAMPLE:
# Test sample: [100, 105, 110, 115, 110, 100, 95, 90]
# 5 nearest neighbors in training set:
#   1. [101, 104, 109, 114, 111, 99, 96, 89] → y=120
#   2. [99, 106, 111, 116, 109, 101, 94, 91] → y=121
#   3. [102, 105, 108, 113, 112, 98, 97, 88] → y=119
#   4. [100, 107, 110, 115, 110, 100, 95, 90] → y=122
#   5. [98, 104, 111, 114, 108, 99, 96, 91] → y=120
# Prediction: mean([120, 121, 119, 122, 120]) = 120.4

y_pred = model.predict(X_test)

# METRICS:
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# RESULT:
# MAE: 1.5146
# RMSE: 2.1837
# ⚠️ Worse than Linear Regression
# Reason: KNN sensitive to outliers, window size small
```

### D.2.3. Model 3: Support Vector Machine (SVM)

```python
from sklearn.svm import SVR

# THEORY:
# Find hyperplane that maximizes margin while fitting data
# Use kernel trick for non-linear boundaries
# Regularization parameter C controls overfitting

model = SVR(kernel='rbf', C=100, epsilon=0.1)
model.fit(X_train, y_train)

# KERNEL: RBF (Radial Basis Function)
# K(x, x') = exp(-gamma * ||x - x'||^2)
# Captures non-linear patterns

y_pred = model.predict(X_test)

# METRICS:
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# RESULT:
# MAE: 0.7946
# RMSE: 2.2987
# ~ Similar to Linear Regression
```

### D.2.4. Model 4: Decision Tree

```python
from sklearn.tree import DecisionTreeRegressor

# THEORY:
# Recursively split data based on features
# Each node: find best split that minimizes MSE
# Leaf node: predict average of samples there

model = DecisionTreeRegressor(max_depth=15, random_state=42)
model.fit(X_train, y_train)

# TREE EXAMPLE:
#           Root
#         /      \
#    If x5 < 105  If x5 >= 105
#     /    \        /    \
#   Leaf  Leaf    Leaf   Leaf
# pred:  pred:   pred:  pred:
#  118    122     125    128

y_pred = model.predict(X_test)

# METRICS:
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# RESULT:
# MAE: 0.0604
# RMSE: 0.2681
# ✓✓ EXCELLENT! Rất tốt
# Reason: Decision tree perfect fit on training data
# ⚠️ Warning: Likely overfitting!
```

### D.2.5. Model 5: Bagging Regressor

```python
from sklearn.ensemble import BaggingRegressor

# THEORY:
# 1. Create multiple subsets of training data (with replacement)
# 2. Train separate model on each subset
# 3. Average predictions (reduce variance)

model = BaggingRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# PROCESS:
# Subset 1: Train model 1
# Subset 2: Train model 2
# ...
# Subset 50: Train model 50
# Final prediction: mean([pred1, pred2, ..., pred50])

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# RESULT:
# MAE: 0.6730
# RMSE: 1.0454
```

### D.2.6. Model 6: Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

# THEORY:
# 1. Build many random decision trees
# 2. At each split, use random subset of features
# 3. Average predictions from all trees
# 4. Reduces overfitting compared to single tree

model = RandomForestRegressor(n_estimators=100, max_depth=15, 
                              random_state=42)
model.fit(X_train, y_train)

# FOREST WITH 100 TREES:
# Tree 1: Uses features [x1, x3, x5, x7]
# Tree 2: Uses features [x2, x4, x6, x8]
# ...
# Tree 100: Uses features [x1, x2, x3, x8]
# Final: mean(all predictions)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# RESULT:
# MAE: 0.6419
# RMSE: 0.9820
# ✓ Good balance between bias & variance
```

### D.2.7. Model 7: Extra Trees (Extremely Randomized Trees)

```python
from sklearn.ensemble import ExtraTreesRegressor

# THEORY:
# Similar to Random Forest but:
# 1. Random split thresholds at each feature
# 2. Fewer computations (faster)
# 3. Different bias-variance tradeoff

model = ExtraTreesRegressor(n_estimators=100, max_depth=15,
                            random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# RESULT:
# MAE: 0.0912
# RMSE: 0.1374
# ✓✓ BEST among ML models!
# Very similar to Decision Tree (both tree-based)
```

### D.2.8. Model 8: AdaBoost Regressor

```python
from sklearn.ensemble import AdaBoostRegressor

# THEORY:
# 1. Train weak learner (small tree)
# 2. Increase weight on misclassified samples
# 3. Train next model on weighted data
# 4. Repeat, then combine all models

model = AdaBoostRegressor(n_estimators=100, learning_rate=0.5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# RESULT:
# MAE: 8.1616
# RMSE: 10.4113
# ❌ WORST! Very bad
# Reason: Not suitable for this regression task
#         Better for classification
```

### D.2.9. Model 9: Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingRegressor

# THEORY:
# 1. Train first weak learner
# 2. Compute residuals (y_true - y_pred)
# 3. Train next learner on residuals
# 4. Update predictions: new_pred = old_pred + learning_rate * residual_pred
# 5. Repeat

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                  max_depth=7)
model.fit(X_train, y_train)

# PROCESS:
# Iteration 1: Tree 1 predicts [120, 121, 119, ...]
# Residuals: y_true - pred = [5, -2, 3, ...]
# Iteration 2: Tree 2 predicts residuals
# Iteration 3-100: Continue...

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# RESULT:
# MAE: 0.4808
# RMSE: 0.6051
# ✓ Very good! Balanced performance
```

### D.2.10. Model 10: XGBoost

```python
import xgboost as xgb

# THEORY:
# Extreme Gradient Boosting
# Similar to Gradient Boosting but:
# 1. More optimized (faster training)
# 2. Regularization (L1, L2)
# 3. Better handling of missing values
# 4. Tree pruning

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1,
                         max_depth=7)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# RESULT:
# MAE: 0.9622
# RMSE: 1.5714
# ✓ Reasonable, but not as good as Gradient Boosting
```

### D.2.11. Model 11: Voting Regressor

```python
from sklearn.ensemble import VotingRegressor

# THEORY:
# 1. Train multiple diverse models
# 2. Average their predictions
# 3. Combines strengths of all models

# Base models
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, max_depth=15)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
et = ExtraTreesRegressor(n_estimators=100, max_depth=15)

# Voting ensemble
model = VotingRegressor(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('gb', gb),
        ('et', et)
    ]
)
model.fit(X_train, y_train)

# PREDICTION PROCESS:
# pred_lr = LinearRegression.predict(X_test) = [120, 121, 119, ...]
# pred_rf = RandomForest.predict(X_test) = [121, 120, 120, ...]
# pred_gb = GradientBoosting.predict(X_test) = [119, 122, 118, ...]
# pred_et = ExtraTrees.predict(X_test) = [120.5, 120.5, 119.5, ...]
# Final: mean([120, 121, 119.5, 120]) = 120.125

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# RESULT:
# MAE: 0.7581
# RMSE: 1.0009
# ✓ Good, combines strengths
```

## D.3. Comparative Performance - 11 Models

```
RANKING BY MAE (Lower is Better):
1. Decision Tree:        0.0604  ✓✓✓
2. Extra Trees:          0.0912  ✓✓✓
3. Gradient Boosting:    0.4808  ✓✓
4. Bagging:              0.6730  ✓
5. Voting Regressor:     0.7581  ✓
6. Linear Regression:    0.7785  ✓
7. SVM:                  0.7946  ✓
8. XGBoost:              0.9622  ✓
9. Random Forest:        0.6419  ✓
10. KNN:                 1.5146  ~
11. AdaBoost:            8.1616  ❌

RANKING BY RMSE (Lower is Better):
1. Decision Tree:        0.2681  ✓✓✓
2. Extra Trees:          0.1374  ✓✓✓
3. Gradient Boosting:    0.6051  ✓✓
... (similar order)

KEY FINDINGS:
├─ Tree-based models (Decision Tree, Extra Trees, GB): Best
├─ Ensemble methods: Good
├─ Linear/SVM: Acceptable
├─ AdaBoost: Not suitable
└─ Decision Tree & Extra Trees: ~0.06 MAE (maybe overfitting?)
```

---

# E. DEEP LEARNING ARCHITECTURE - ULTRA CHI TIẾT

## E.1. 3 Input Channels - Cụ Thể Chi Tiết

### E.1.1. Channel 1: Sequence Input

```
NAME: seq_in
SHAPE: (batch_size, 20)
DTYPE: float32

CONTENT:
├─ Last 20 values of NORMALIZED imputed sequence
├─ Normalization: MinMaxScaler([0, 1])
└─ Purpose: Provide raw temporal pattern

EXAMPLE DATA (for 1 sample):
seq_in = [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, ..., 0.35]
                                       └─ 20 values
         (normalized 0-1 range)

WHAT THE MODEL LEARNS FROM THIS:
├─ Trend: Increasing from 0.12 to 0.35
├─ Acceleration: Slope changes over time
├─ Patterns: Subtle variations (seasonality)
└─ Direct signal: The sequence itself
```

### E.1.2. Channel 2: Reference Input

```
NAME: ref_in
SHAPE: (batch_size, 20)
DTYPE: float32

CONTENT:
├─ Last 20 values of NORMALIZED reference (Waterlevel)
├─ Normalization: MinMaxScaler([0, 1])
└─ Purpose: Provide comparison/context

EXAMPLE DATA:
ref_in = [0.11, 0.14, 0.17, 0.19, 0.23, 0.24, ..., 0.34]
         (same timeframe as seq_in)

COMPARISON:
seq_in vs ref_in:
├─ seq_in: [0.12, 0.15, 0.18, 0.20, 0.22, 0.25, ..., 0.35]
├─ ref_in: [0.11, 0.14, 0.17, 0.19, 0.23, 0.24, ..., 0.34]
└─ Difference: Typically small (both from same source/period)

WHAT THE MODEL LEARNS:
├─ Comparison: How different imputed is from reference
├─ Error signal: Helps correct imputation bias
├─ Alignment: Whether trends match
└─ Validation: Signal about confidence
```

### E.1.3. Channel 3: Context Input

```
NAME: ctx_in
SHAPE: (batch_size, context_dim)
       where context_dim = 8-12 (depends on window)
DTYPE: float32

CONTENT (from eDTWBI extraction):
└─ context_feature = [left_1, left_2, left_3,
                      right_1, right_2, right_3,
                      mean_gap, std_gap]

EXAMPLE DATA (context_dim=8):
ctx_in = [0.10, 0.12, 0.15,     # left context (3 values)
          0.30, 0.32, 0.35,     # right context (3 values)
          0.22,                  # mean of best gaps
          0.08]                  # std of best gaps

MEANING OF EACH COMPONENT:
1. Left context [0.10, 0.12, 0.15]:
   └─ Behavior BEFORE the gap (normalized)
   
2. Right context [0.30, 0.32, 0.35]:
   └─ Behavior AFTER the gap (normalized)
   
3. Mean of best gaps [0.22]:
   └─ Average value of best-matching gap candidates
   └─ Indicates: What values are typical for this gap?
   
4. Std of best gaps [0.08]:
   └─ Standard deviation of top-k candidates
   └─ Indicates: How consistent are the candidates?
   └─ Low std: Confident imputation
   └─ High std: Uncertain imputation

WHAT THE MODEL LEARNS:
├─ Pattern quality: From mean & std
├─ Confidence: Low std = high confidence
├─ Context: Surrounding behavior patterns
├─ Prior: From eDTWBI (strong signal)
└─ Decision: Refine or trust eDTWBI
```

## E.2. Reshape & Concatenate Operations

```python
# INPUT SHAPES BEFORE RESHAPE:
seq_in shape:     (batch, 20)        # 1D array
ref_in shape:     (batch, 20)        # 1D array

# RESHAPE OPERATION:
seq_r = Reshape((20, 1))(seq_in)
ref_r = Reshape((20, 1))(ref_in)

# SHAPES AFTER RESHAPE:
seq_r shape:      (batch, 20, 1)     # 2D: 20 timesteps, 1 feature
ref_r shape:      (batch, 20, 1)     # 2D: 20 timesteps, 1 feature

# WHY RESHAPE?
├─ LSTM/GRU expects 3D input: (batch, timesteps, features)
├─ seq_in (batch, 20) is 2D
├─ Reshape to (batch, 20, 1) makes it 3D
└─ 1 feature per timestep (the sequence value)

# CONCATENATE OPERATION:
merged_seq = Concatenate(axis=-1)([seq_r, ref_r])

# CONCATENATION DETAILS:
seq_r shape:        (batch, 20, 1)
ref_r shape:        (batch, 20, 1)
axis=-1 means:      Concatenate along last axis (features)

# RESULT:
merged_seq shape:   (batch, 20, 2)
                    └─ 20 timesteps, 2 features per timestep

# FOR EACH TIMESTEP t:
merged_seq[:, t, :] = [seq_r[:, t, 0], ref_r[:, t, 0]]
                    = [sequence value, reference value]

# EXAMPLE:
seq_r[:, 0, :] = [0.12]
ref_r[:, 0, :] = [0.11]
merged_seq[:, 0, :] = [0.12, 0.11]

# VISUAL:
Before:  seq_r        ref_r        After:  merged_seq
timestep  value       value       timestep  (seq, ref)
0         0.12        0.11    →   0         (0.12, 0.11)
1         0.15        0.14    →   1         (0.15, 0.14)
2         0.18        0.17    →   2         (0.18, 0.17)
...       ...         ...    →   ...       ...
19        0.35        0.34    →   19        (0.35, 0.34)
```

## E.3. Bidirectional LSTM - Chi Tiết Hoàn Chỉnh

### E.3.1. Standard LSTM vs Bidirectional LSTM

```
STANDARD LSTM (UNIDIRECTIONAL):
    Input: (batch, 20, 2)
    ↓
    [LSTM layer: forward only]
    ↓
    Output: (batch, 20, 32) if return_sequences=True
            (batch, 32) if return_sequences=False
    
    Processing: t=0 → t=1 → t=2 → ... → t=19
    └─ Only uses past information (left-to-right)

BIDIRECTIONAL LSTM:
    Input: (batch, 20, 2)
    ↙ ↘
Forward LSTM    Backward LSTM
    ↓               ↓
    (batch, 20, 32)  (batch, 20, 32)
    ↓               ↓
    t=0→19          t=19→0 (reverse)
    ↙ ↘
    Concatenate
    ↓
    Output: (batch, 20, 64) if return_sequences=True
            (batch, 64) if return_sequences=False
            
    └─ Uses both past AND future information!
```

### E.3.2. Forward & Backward LSTM Processing

```
FORWARD LSTM (Left to Right):
    t=0: h_fwd[0] = LSTM(x[0], h_fwd[-1])
    t=1: h_fwd[1] = LSTM(x[1], h_fwd[0])
    t=2: h_fwd[2] = LSTM(x[2], h_fwd[1])
    ...
    t=19: h_fwd[19] = LSTM(x[19], h_fwd[18])
    
    Output: h_fwd[0], h_fwd[1], ..., h_fwd[19]
    └─ Each uses information from t=0 to t
    └─ h_fwd[19] contains full sequence info (best)

BACKWARD LSTM (Right to Left):
    t=19: h_bwd[19] = LSTM(x[19], h_bwd[20])
    t=18: h_bwd[18] = LSTM(x[18], h_bwd[19])
    t=17: h_bwd[17] = LSTM(x[17], h_bwd[18])
    ...
    t=0: h_bwd[0] = LSTM(x[0], h_bwd[1])
    
    Output (in order): h_bwd[0], h_bwd[1], ..., h_bwd[19]
    └─ Each uses information from t=19 to t
    └─ h_bwd[0] contains full sequence info (best)

CONCATENATION AT EACH TIMESTEP:
    At t=0: concat([h_fwd[0], h_bwd[0]])
            = concat([future info from 0→1→...→19,
                     future info from 0←1←...←19])
            = (32 + 32 = 64 dims)

    At t=10: concat([h_fwd[10], h_bwd[10]])
             = concat([past 0-10 + future 10-19])
             = (64 dims - most informative!)

    At t=19: concat([h_fwd[19], h_bwd[19]])
             = concat([all past info, future 19])
             = (64 dims)
```

### E.3.3. LSTM Layer 1: return_sequences=True

```python
# LAYER 1:
x = Bidirectional(LSTM(units=32, return_sequences=True))(merged_seq)

# INPUT:
merged_seq shape: (batch, 20, 2)
                  └─ 20 timesteps, 2 features

# LSTM PARAMETERS:
units = 32
└─ Each LSTM cell outputs 32-dim vector

# return_sequences=True
└─ Return output at EVERY timestep
└─ NOT just the last output

# OUTPUT SHAPE:
x shape: (batch, 20, 64)
         └─ 20 timesteps (same as input!)
         └─ 64 features = 32 forward + 32 backward

# OUTPUT CONTENT:
x[0] = concat(forward_LSTM[0], backward_LSTM[0]) = 64-dim vector
x[1] = concat(forward_LSTM[1], backward_LSTM[1]) = 64-dim vector
...
x[19] = concat(forward_LSTM[19], backward_LSTM[19]) = 64-dim vector

# WHY return_sequences=True?
├─ We want output at each timestep
├─ Later, we apply another LSTM on top
├─ So we need (batch, 20, 64) not (batch, 64)
└─ The second LSTM processes all 20 timesteps
```

### E.3.4. Dropout(0.2)

```python
x = Dropout(0.2)(x)

# WHAT DROPOUT DOES:
├─ Randomly set 20% of values to 0 during training
├─ Scale remaining 80% by 1/0.8 = 1.25
└─ During inference: NO dropout, use all weights

# EFFECT:
Before dropout: x shape (batch, 20, 64)
                values: [0.5, -0.3, 0.8, ..., -0.2]

After dropout (training):
                values: [0.5*1.25, 0, 0.8*1.25, ..., 0]
                        └─ ~20% zeroed

During inference (test):
                No dropout applied
                Use full x

# WHY DROPOUT?
├─ Prevent overfitting
├─ Act as regularization (ensemble-like)
├─ Reduce co-adaptation of neurons
└─ Improve generalization
```

### E.3.5. LSTM Layer 2: return_sequences=False

```python
x = Bidirectional(LSTM(units=32))(x)

# INPUT:
x shape: (batch, 20, 64)
         └─ 20 timesteps, 64 features

# LSTM PROCESSING:
Forward LSTM:
  t=0: h_f[0] = LSTM(x[0, :], h_f[-1])  ← input: 64-dim
  t=1: h_f[1] = LSTM(x[1, :], h_f[0])
  ...
  t=19: h_f[19] = LSTM(x[19, :], h_f[18])  ← final hidden state

Backward LSTM:
  t=19: h_b[19] = LSTM(x[19, :], h_b[20])
  ...
  t=0: h_b[0] = LSTM(x[0, :], h_b[1])  ← final hidden state

# RETURN (return_sequences=False):
Only return LAST output (concatenated):
x = concat([h_f[19], h_b[0]])
  = concat([forward final, backward final])
  = 64-dim vector

# OUTPUT SHAPE:
x shape: (batch, 64)
         └─ 64 = 32 forward + 32 backward
         └─ Collapsed from (batch, 20, 64) to (batch, 64)

# WHY return_sequences=False?
├─ We only care about final representation
├─ This representation should encode the full sequence
├─ Simplifies downstream processing
└─ h_f[19] has info from entire forward pass
└─ h_b[0] has info from entire backward pass
```

## E.4. Context Branch Processing

```python
# INPUT:
ctx_in shape: (batch, context_dim=12)
              └─ 12-dim vector from eDTWBI

# LAYER 1: Dense
ctx_branch = Dense(32, activation='relu')(ctx_in)

# Dense layer computation:
ctx_branch = relu(ctx_in @ W + b)
where:
  W: (12, 32) weight matrix
  b: (32,) bias vector
  @: matrix multiplication

# OUTPUT:
ctx_branch shape: (batch, 32)
                 └─ Project from 12 → 32 dims

# ACTIVATION (ReLU):
relu(x) = max(0, x)
└─ Non-linearity, captures complex patterns

# LAYER 2: Dropout
ctx_branch = Dropout(0.2)(ctx_branch)

# OUTPUT:
ctx_branch shape: (batch, 32)
                 └─ (same shape, but ~20% zeroed in training)
```

## E.5. Fusion & Final Layers

```python
# CONCATENATE SEQ BRANCH + CONTEXT BRANCH:
concat = Concatenate()([x, ctx_branch])

# BEFORE CONCATENATE:
x shape:         (batch, 64)  ← from BiLSTM
ctx_branch shape: (batch, 32)  ← from Dense + Dropout

# AFTER CONCATENATE:
concat shape: (batch, 96)
            = (batch, 64 + 32)

# LAYER 1: Dense + Dropout
z = Dense(32, activation='relu')(concat)
# (batch, 96) → (batch, 32)

z = Dropout(0.1)(z)
# Dropout 10% (lower than before, final stage)
# (batch, 32) → (batch, 32)

# LAYER 2: Output Dense
out = Dense(1)(z)  # No activation (Linear)
# (batch, 32) → (batch, 1)

# FINAL OUTPUT:
out shape: (batch, 1)
out values: [0.34, 0.28, 0.41, ...]
└─ Normalized predictions [0, 1]
└─ Each value is imputed normalized value
```

---

# F. TRAINING & OPTIMIZATION - CHI TIẾT CẶN KẼ

## F.1. Callbacks Configuration

### F.1.1. EarlyStopping Chi Tiết

```python
callbacks.append(
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
)

# PARAMETER MEANINGS:

1. monitor='val_loss'
   ├─ Watch validation loss
   ├─ Epoch result should be: val_loss = 0.0523
   └─ If this metric doesn't improve, consider stopping

2. patience=10
   ├─ Wait 10 epochs without improvement
   ├─ Example:
   │  Epoch 15: val_loss = 0.0620 (BEST so far)
   │  Epoch 16: val_loss = 0.0625 (no improve) → count=1
   │  Epoch 17: val_loss = 0.0628 (no improve) → count=2
   │  ...
   │  Epoch 25: val_loss = 0.0750 (no improve) → count=10
   │  Epoch 26: STOP! (count reached patience)
   └─ Restore weights from Epoch 15 (best val_loss)

3. restore_best_weights=True
   ├─ After stopping, load the best weights
   ├─ Not the final weights (which might be worse)
   └─ Use model from Epoch 15 (lowest val_loss)

4. verbose=1
   ├─ Print messages
   ├─ Output:
   │  Epoch 26/50: EarlyStopping: Stop training
   │              Restoring model weights from the epoch
   │              with the best validation loss
   └─ Helps track what's happening
```

### F.1.2. ReduceLROnPlateau Chi Tiết

```python
callbacks.append(
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
)

# PARAMETER MEANINGS:

1. monitor='val_loss'
   └─ Watch validation loss

2. factor=0.5
   ├─ Multiply learning rate by 0.5
   ├─ If LR was 0.001, new LR = 0.001 * 0.5 = 0.0005
   └─ Reduce learning rate to half

3. patience=3
   ├─ Wait 3 epochs without improvement before reducing LR
   └─ After 3 epochs of no progress, reduce LR

4. EXAMPLE EXECUTION:

Epoch 1-5: LR = 0.001
  - Epoch 1: val_loss = 0.2500 (BEST)
  - Epoch 2: val_loss = 0.1800 (BEST)
  - Epoch 3: val_loss = 0.1200 (BEST)
  - Epoch 4: val_loss = 0.0800 (BEST)
  - Epoch 5: val_loss = 0.0600 (BEST)

Epoch 6-8: Still LR = 0.001, but no improvement
  - Epoch 6: val_loss = 0.0620 (no improve) → count=1
  - Epoch 7: val_loss = 0.0630 (no improve) → count=2
  - Epoch 8: val_loss = 0.0635 (no improve) → count=3 → count=patience!

Epoch 9 onwards: LR REDUCED to 0.0005
  - Epoch 9: val_loss = 0.0580 (IMPROVE!) ✓ count reset to 0
  - Epoch 10: val_loss = 0.0550 (IMPROVE!) ✓
  - Epoch 11-13: No improve, count=1,2,3

Epoch 14 onwards: LR REDUCED again to 0.00025
  - Continue training with smaller LR...

# WHY ReduceLROnPlateau?
├─ When gradient becomes small, updates slow
├─ Smaller LR = finer adjustments
├─ Helps escape local plateaus
└─ Improves convergence
```

## F.2. Training Loop

```python
model = build_model('LSTM', sequence_length=20, context_dim=12, units=32)

history = model.fit(
    [X_seq_tr, X_ref_tr, X_ctx_tr],  # Inputs
    y_tr,                             # Target
    validation_split=0.15,            # 15% of training for validation
    epochs=50,                        # Maximum 50 epochs
    batch_size=128,                   # 128 samples per batch
    callbacks=callbacks,              # EarlyStopping + ReduceLROnPlateau
    verbose=2                         # Detailed output
)

# TRAINING PROCESS:

Epoch 1/50
----------
Batch 1/176: loss = 0.2541, mae = 0.4521
Batch 2/176: loss = 0.2304, mae = 0.4200
...
Batch 176/176: loss = 0.1800, mae = 0.3900

Training loss:   0.1950
Validation loss: 0.1823 (BEST!)
Learning rate:   0.001000

Epoch 2/50
----------
Training loss:   0.1234
Validation loss: 0.1156 (BEST!)
Learning rate:   0.001000

...

Epoch 15/50
----------
Training loss:   0.0023
Validation loss: 0.0052 (BEST!)
Learning rate:   0.001000

Epoch 16-18/50
----------
Validation loss increases (no improvement)
LR reduced to 0.0005

Epoch 25/50
----------
Training loss:   0.0005
Validation loss: 0.0048 (no improve vs Epoch 15)
count = 10 (patience reached!)

EarlyStopping: Restoring best model weights from Epoch 15
Training stopped!

# HISTORY OBJECT:
history.history = {
    'loss': [0.195, 0.123, 0.089, ..., 0.0005],
    'mae': [0.384, 0.301, 0.224, ..., 0.0041],
    'val_loss': [0.1823, 0.1156, 0.0898, ..., 0.0048],
    'val_mae': [0.3956, 0.2987, 0.2234, ..., 0.0045]
}
```

## F.3. Mixed Precision Training

```python
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# WHAT IS MIXED PRECISION?
├─ Forward pass: Use float16 (16-bit floating point)
│  ├─ Faster computation
│  ├─ Less memory usage
│  └─ Less accurate but usually fine
│
└─ Loss computation: Use float32 (32-bit floating point)
   ├─ Better precision for gradients
   ├─ Avoids numerical instability
   └─ Slow but critical

# PERFORMANCE GAINS:
├─ Speed: 1.5-2x faster on modern GPUs
├─ Memory: 50% reduction in memory usage
├─ Accuracy: Usually same as float32
└─ Suitable for: Batch size, learning, general training

# TRADEOFFS:
├─ ✓ Faster, uses less memory
├─ ❌ Potential numerical issues (rare with modern GPUs)
└─ → Use for Deep Learning, not for critical financial calculations
```

---

# G. METRICS & EVALUATION - CẶN KẼ

(Tiếp tục với phần H-K...)

---

**Tài liệu này quá dài (>10,000 từ), tôi sẽ tạo một file mới cho phần còn lại...**
