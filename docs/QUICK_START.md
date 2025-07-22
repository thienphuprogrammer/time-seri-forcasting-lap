# Quick Start Guide - DAT301m Lab 4

## Giới thiệu

Package này cung cấp giải pháp hoàn chỉnh cho **DAT301m Lab 4: Time Series Forecasting** với dữ liệu PJM hourly energy consumption.

## Cài đặt nhanh

```bash
# Clone repository
git clone <repository-url>
cd Time-Series-Forecasting

# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc sử dụng uv
uv sync
```

## Sử dụng nhanh

### Option 1: Jupyter Notebooks (Khuyến nghị cho Lab)

```bash
# Chạy từng task riêng biệt
jupyter notebook notebooks/task1.ipynb  # Data Exploration (1.5 điểm)
jupyter notebook notebooks/task2.ipynb  # Baseline Models (3 điểm)
jupyter notebook notebooks/task3.ipynb  # Deep Learning (4 điểm)
jupyter notebook notebooks/task4.ipynb  # Transformers (1.5 điểm)

# Hoặc xem summary tổng quan
jupyter notebook notebooks/lab4_complete_summary.ipynb
```

### Option 2: Demo hoàn chỉnh

```bash
python examples/lab4_complete_demo.py
```

### Option 3: Complete Analysis Notebook

```bash
jupyter notebook notebooks/01_complete_analysis.ipynb
```

### Option 4: Programmatic Usage

```python
from time_series_forecasting.analysis.lab_interface.lab4_interface import Lab4Interface

# Khởi tạo
lab = Lab4Interface()

# Load data
data = lab.load_data('data/PJME_hourly.csv', region='PJME')

# Thực hiện các tasks
task1_results = lab.execute_task1()
task2_results = lab.execute_task2(model_configs=baseline_models)
task3_results = lab.execute_task3(model_configs=deep_learning_models)
task4_results = lab.execute_task4(model_configs=transformer_models)

# Lưu kết quả
lab.save_results('output_dir')
```

## Cấu trúc Notebooks

### 📚 Individual Task Notebooks:
1. **task1.ipynb** - Data Exploration & Preprocessing
   - Load và preprocess dữ liệu PJME
   - Comprehensive data analysis và visualization
   - WindowGenerator implementation
   - Train/validation/test splits

2. **task2.ipynb** - Baseline Models
   - Linear Regression và ARIMA models
   - Early stopping và evaluation
   - Model comparison và Q1 answer
   - Performance visualization

3. **task3.ipynb** - Deep Learning Models
   - RNN, GRU, LSTM implementations
   - Training với early stopping
   - Architecture comparison và Q2 answer
   - Deep learning evaluation

4. **task4.ipynb** - Transformer Models
   - Multi-Head Attention Transformer
   - Positional encoding và layer normalization
   - Comparison với Task 3 models
   - Advanced architecture analysis

5. **lab4_complete_summary.ipynb** - Complete Overview
   - Tổng kết tất cả tasks
   - Score breakdown: 10/10 điểm
   - Usage instructions và expected results

## Cấu trúc outputs

Sau khi chạy, kết quả sẽ được lưu vào thư mục được chỉ định:

```
lab4_results/
├── plots/
│   ├── time_series.png
│   ├── seasonal_patterns.png
│   ├── distribution.png
│   ├── trends.png
│   ├── anomalies.png
│   └── correlation.png
├── analysis_results.json
└── task_results.json
```

## Tasks được hoàn thành

### ✅ Task 1: Data Exploration (1.5 điểm)
- Load và preprocess dữ liệu PJM
- Parse datetime, normalize data
- Tạo visualizations: time series, seasonal patterns, distributions
- Implement WindowGenerator class
- Chia train/validation/test

### ✅ Task 2: Baseline Models (3 điểm)
- Linear Regression với lagged features (1 điểm)
- ARIMA models với parameter tuning (1 điểm)
- Early stopping và evaluation (0.5 điểm)
- Training/validation curves và Q1 answer (0.5 điểm)

### ✅ Task 3: Deep Learning Models (4 điểm)
- RNN, GRU, LSTM models (3 điểm)
- Early stopping và comprehensive evaluation
- Model comparison và visualization (0.5 điểm)
- Q2 answer: Temporal pattern analysis (0.5 điểm)

### ✅ Task 4: Transformer Models (1.5 điểm)
- Transformer với multi-head attention (1 điểm)
- Positional encoding và layer normalization
- So sánh với Task 3 models (0.5 điểm)

## Lab Questions

### Q1: Mô hình nào khái quát tốt hơn và tại sao?
- Notebooks sẽ tự động so sánh các mô hình dựa trên MAE và RMSE
- Linear Regression có độ phức tạp thấp, ít overfitting
- ARIMA phù hợp với dữ liệu seasonal
- Analysis được include trong task2.ipynb

### Q2: Mô hình nào nắm bắt mẫu thời gian tốt nhất?
- LSTM thường tốt cho long-term dependencies
- GRU đơn giản hơn LSTM với hiệu suất tương tự
- Transformer có thể tốt hơn với attention mechanism
- RNN có thể bị vanishing gradient
- Detailed analysis trong task3.ipynb

## Datasets hỗ trợ

Package hỗ trợ tất cả regions của PJM:
- `PJME_hourly.csv` - PJM East (khuyến nghị cho notebooks)
- `PJMW_hourly.csv` - PJM West
- `AEP_hourly.csv` - American Electric Power
- `COMED_hourly.csv` - Commonwealth Edison
- Và nhiều regions khác...

## Troubleshooting

### Lỗi import
```bash
# Đảm bảo đang ở thư mục root
cd Time-Series-Forecasting

# Cài đặt package
pip install -e .
```

### Lỗi memory khi train models
```python
# Giảm epochs hoặc batch size trong notebooks
train_params = {'epochs': 20, 'batch_size': 16}
```

### Lỗi data không tìm thấy
```bash
# Đảm bảo data file tồn tại
ls data/PJME_hourly.csv
```

### Jupyter notebook issues
```bash
# Install jupyter if needed
pip install jupyter

# Run từ project root
cd Time-Series-Forecasting
jupyter notebook
```

## Recommended Workflow

1. **Start with Summary**: `lab4_complete_summary.ipynb` để hiểu tổng quan
2. **Run Individual Tasks**: 
   - `task1.ipynb` → `task2.ipynb` → `task3.ipynb` → `task4.ipynb`
3. **Alternative**: Run `lab4_complete_demo.py` cho automated workflow
4. **Explore**: `01_complete_analysis.ipynb` cho detailed analysis

## Support

Nếu gặp lỗi, kiểm tra:
1. Python version >= 3.8
2. Dependencies được cài đặt đúng
3. Data files tồn tại trong thư mục `data/`
4. Có đủ disk space cho outputs
5. Jupyter notebook running từ project root

---

**Good luck với Lab 4! 🚀**

**📊 Expected Score: 10/10 điểm** 