# DAT301m Lab 4: Time Series Forecasting

A comprehensive time series forecasting solution for PJM hourly energy consumption data, designed to complete all requirements for DAT301m Lab 4.

## 🎯 Lab Requirements Fulfilled

### Task 1: Dataset Exploration and Preprocessing (1.5 points)
- ✅ Data loading and parsing with automatic datetime formatting
- ✅ Data normalization and missing value handling
- ✅ Comprehensive visualizations (time series, seasonal decomposition, distributions)
- ✅ WindowGenerator class for configurable input/output windows
- ✅ Proper train/validation/test splits

### Task 2: Baseline Models (3 points)
- ✅ Linear Regression with lagged features
- ✅ ARIMA and SARIMA models
- ✅ Early stopping and proper evaluation
- ✅ Training/validation curves and forecast plots
- ✅ Q1 answer: Model comparison and overfitting analysis

### Task 3: Deep Learning Models (4 points)
- ✅ Recurrent models: RNN, GRU, LSTM
- ✅ CNN-LSTM hybrid model
- ✅ Ensemble model combining multiple approaches
- ✅ Multi-step forecasting (24 hours ahead)
- ✅ Q2 answer: Temporal pattern analysis

### Task 4: Advanced Models (1.5 points)
- ✅ Seq2Seq with Attention mechanism
- ✅ Transformer models with positional encoding
- ✅ Multi-head attention and layer normalization
- ✅ Comparison with Task 3 models

## 🚀 Quick Start

### Option 1: Complete Workflow (Recommended)
```python
from time_series_forecasting.analysis.lab4_interface import DAT301mLab4Interface

# Initialize
lab = DAT301mLab4Interface(
    data_path='data/PJME_hourly.csv',
    region='PJME',
    input_width=24,
    label_width=1,
    shift=1
)

# Run complete lab workflow
results = lab.run_complete_lab(
    output_dir='lab4_results/',
    save_plots=True,
    multi_step=True,
    create_ensemble=True
)
```

### Option 2: Interactive Notebook
1. Open `notebooks/01_complete_analysis.ipynb`
2. Run all cells to complete the entire lab
3. All outputs will be saved automatically

### Option 3: Individual Tasks
```python
# Task 1: Data exploration
task1_results = lab.execute_task1(
    datetime_col='Datetime',
    target_col='PJME_MW',
    create_plots=True
)

# Task 2: Baseline models
task2_results = lab.execute_task2(
    epochs=100,
    patience=10
)

# Task 3: Deep learning models
task3_results = lab.execute_task3(
    epochs=100,
    units=64,
    layers=2,
    create_ensemble=True,
    multi_step=True
)

# Task 4: Transformer models
task4_results = lab.execute_task4(
    epochs=100,
    num_heads=8,
    d_model=128,
    num_layers=4
)
```

## 📊 Available Datasets

The package supports all PJM regions:
- `AEP_hourly.csv` - American Electric Power
- `COMED_hourly.csv` - Commonwealth Edison
- `DAYTON_hourly.csv` - Dayton Power & Light
- `DEOK_hourly.csv` - Duke Energy Ohio/Kentucky
- `DOM_hourly.csv` - Dominion Virginia Power
- `DUQ_hourly.csv` - Duquesne Light
- `EKPC_hourly.csv` - East Kentucky Power
- `FE_hourly.csv` - FirstEnergy
- `NI_hourly.csv` - Northern Illinois Hub
- `PJME_hourly.csv` - PJM East
- `PJMW_hourly.csv` - PJM West
- `PJM_Load_hourly.csv` - Total PJM Load

## 🏗️ Architecture

```
time_series_forecasting/
├── analysis/
│   ├── lab4_interface.py      # Main interface for lab completion
│   └── pjm_analyzer.py        # PJM-specific analysis tools
├── core/
│   ├── data_processor.py      # Data loading and preprocessing
│   └── window_generator.py    # Time series window generation
├── models/
│   ├── model_factory.py       # All model implementations
│   └── model_trainer.py       # Training and evaluation
├── pipeline/
│   └── forecasting_pipeline.py # Complete workflow orchestration
└── utils/
    └── logger.py              # Logging utilities
```

## 🎨 Features

### Data Processing
- Automatic datetime parsing and timezone handling
- Multiple normalization methods (MinMax, Standard, Robust)
- Missing data handling (interpolation, forward-fill, backward-fill)
- Interactive visualizations with Plotly

### Model Library
- **Baseline**: Linear Regression, ARIMA, SARIMA
- **Deep Learning**: RNN, GRU, LSTM, CNN-LSTM
- **Advanced**: Seq2Seq with Attention, Transformer
- **Ensemble**: Model combination with multiple strategies

### Evaluation
- Comprehensive metrics: MAE, RMSE, R², MAPE
- Training/validation curves
- Prediction vs actual plots
- Model comparison visualizations

## 📈 Results and Reports

The system automatically generates:
- **Comprehensive Report**: Complete analysis with all results
- **Visualizations**: Time series plots, model comparisons, forecasts
- **Question Answers**: Detailed responses to Q1 and Q2
- **Performance Metrics**: Full comparison table

## 🔧 Installation

```bash
# Install required packages
pip install -r requirements.txt

# Or using uv (recommended)
uv pip install -r requirements.txt
```

## 📝 Usage Examples

### Basic Usage
```python
from time_series_forecasting import DAT301mLab4Interface

# Initialize with your data
lab = DAT301mLab4Interface('data/PJME_hourly.csv', region='PJME')

# Complete all tasks
results = lab.run_complete_lab()
```

### Custom Configuration
```python
# Advanced configuration
lab = DAT301mLab4Interface(
    data_path='data/COMED_hourly.csv',
    region='COMED',
    input_width=48,      # 48 hours input
    label_width=24,      # 24 hours prediction
    shift=1,             # 1 hour shift
    random_seed=42
)

# Custom training parameters
task3_results = lab.execute_task3(
    epochs=200,
    patience=15,
    units=128,
    layers=3,
    dropout=0.3,
    create_ensemble=True,
    multi_step=True,
    multi_step_horizon=48
)
```

## 📊 Model Performance

The system provides detailed performance analysis:
- Training/validation loss curves
- Early stopping analysis
- Cross-model comparison
- Temporal pattern visualization
- Multi-step forecasting accuracy

## 🎯 Grading Alignment

### Task 1 (1.5 pts)
- ✅ Data cleaning and parsing (0.5 pts)
- ✅ Visualization and analysis (0.5 pts)
- ✅ WindowGenerator implementation (0.5 pts)

### Task 2 (3 pts)
- ✅ Linear Regression (1 pt)
- ✅ ARIMA/SARIMA (1 pt)
- ✅ Evaluation and plots (0.5 pts)
- ✅ Q1 answer (0.5 pts)

### Task 3 (4 pts)
- ✅ RNN/GRU/LSTM (3 pts)
- ✅ Evaluation and plots (0.5 pts)
- ✅ Q2 answer (0.5 pts)
- ✅ Multi-step forecasting (1 pt bonus)

### Task 4 (1.5 pts)
- ✅ Attention/Transformer (1 pt)
- ✅ Comparison analysis (0.5 pts)

## 🤝 Contributing

This is an educational project for DAT301m Lab 4. Feel free to:
- Add new model architectures
- Improve visualization methods
- Enhance evaluation metrics
- Add more PJM regions

## 📄 License

MIT License - Feel free to use for educational purposes.

## 🙏 Acknowledgments

- PJM Interconnection for providing the energy consumption data
- TensorFlow and scikit-learn teams for the ML frameworks
- The open-source community for the supporting libraries

---

**Ready to achieve full marks on DAT301m Lab 4? Just run the notebook and watch the magic happen! 🎉**
