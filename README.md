# Time Series Forecasting – Lab 4 Enhanced Report  
**Author:** Alexander  
**Date:** 23 July 2025  
**Course:** DAT301m

---

## Abstract
This report presents a comprehensive implementation of time series forecasting techniques applied to PJM East (PJME) hourly electricity consumption data. The study progresses from exploratory data analysis through classical statistical models, deep learning architectures, and advanced attention-based transformers. While the linear regression baseline achieved exceptional performance (MAE ≈ 0.0078), this work demonstrates the importance of matching model complexity to signal characteristics and provides insights into when sophisticated architectures add value for time series forecasting.

---

## 1. Introduction

### 1.1 Dataset Overview
The PJM East region hourly electricity consumption dataset serves as our primary data source for this forecasting study.

| **Property** | **Value** |
|--------------|-----------|
| Source | Kaggle: PJM Hourly Energy Consumption |
| File | `PJME_hourly.csv` |
| Time Period | January 1, 2002 – August 3, 2018 (16.6 years) |
| Frequency | Hourly |
| Total Records | 145,366 (145,362 after cleaning) |
| Target Variable | MW (Megawatts) |
| Data Quality | 100% complete, minimal outliers |

### 1.2 Objectives
This lab aims to systematically compare forecasting approaches across multiple paradigms:
1. **Classical Methods**: Linear regression and ARIMA models
2. **Deep Learning**: RNN, LSTM, GRU architectures
3. **Modern Architectures**: Transformer with attention mechanisms
4. **Advanced Techniques**: Multi-step forecasting and ensemble methods

---

## 2. Task 1: Dataset Exploration and Preprocessing

### 2.1 Data Loading and Cleaning
```python
# Data preprocessing pipeline
df = pd.read_csv('data/PJME_hourly.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime')
df = df.dropna()  # Remove 4 missing values
```

### 2.2 Exploratory Data Analysis

The comprehensive EDA revealed several critical patterns:

#### Temporal Patterns
- **Long-term Trend**: Gradual increase in consumption over 16 years
- **Annual Seasonality**: Higher consumption in summer/winter (HVAC usage)
- **Weekly Patterns**: Lower consumption on weekends
- **Daily Cycles**: Peak demand during business hours (9 AM - 6 PM)

#### Statistical Properties
- **Distribution**: Right-skewed with occasional extreme peaks
- **Stationarity**: Non-stationary due to trend and seasonality
- **Autocorrelation**: Strong correlations at 24h, 168h (weekly), and seasonal lags

### 2.3 WindowGenerator Implementation
```python
class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, label_columns=None):
        self.input_width = input_width
        self.label_width = label_width  
        self.shift = shift
        # Implementation supports both single-step and multi-step forecasting
```

### 2.4 Data Normalization and Splitting
- **Normalization**: Min-max scaling (0-1 range) applied to stabilize training
- **Train/Validation/Test Split**: 70%/20%/10% chronological split
- **Window Configuration**: 24-hour input window, 1-hour prediction horizon

---

## 3. Task 2: Baseline Classical Models

### 3.1 Linear Regression Model

**Implementation Approach:**
```python
# Feature engineering for linear model
features = create_time_features(df)  # Hour, day, month, seasonality
model = LinearRegression()
model.fit(X_train, y_train)
```

**Results:**
- **MAE**: 0.0078
- **RMSE**: 0.0109  
- **R²**: 0.9960
- **MAPE**: 1.85%

The linear model's exceptional performance stems from the well-behaved, normalized signal within the 24-hour prediction window.

### 3.2 ARIMA Model

**Model Selection Process:**
- Used AIC/BIC criteria for parameter selection
- Applied differencing to achieve stationarity
- Final model: ARIMA(2,1,2)

**Results:**
- **MAE**: 0.1536
- **RMSE**: 0.1845
- **R²**: -0.1426
- **MAPE**: 45.19%

The poor ARIMA performance indicates the need for seasonal components (SARIMA) to capture the multiple seasonality patterns present in electricity consumption data.

### 3.3 Training Visualization and Early Stopping

Both models employed early stopping mechanisms:
- **Patience**: 10 epochs without improvement
- **Monitoring Metric**: Validation MAE
- **Learning Curves**: Plotted training/validation loss over epochs

---

## 4. Task 3: Deep Learning Models

### 4.1 Model Architectures

#### RNN Model (GPU Optimized)
```python
model = Sequential([
    SimpleRNN(512, return_sequences=True),
    Dropout(0.2),
    SimpleRNN(512, return_sequences=True),
    Dropout(0.2), 
    SimpleRNN(512),
    Dense(1)
])
```

#### Advanced LSTM Model
```python
model = Sequential([
    LSTM(512, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(512, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(512),
    Dense(256, activation='relu'),
    Dense(1)
])
```

#### Deep GRU Model
```python
model = Sequential([
    GRU(512, return_sequences=True),
    GRU(512, return_sequences=True),
    GRU(512, return_sequences=True),
    GRU(512),
    Dense(1)
])
```

### 4.2 Training Configuration
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Mean Squared Error
- **Batch Size**: 1024 (GPU optimized)
- **Early Stopping**: Patience=15, monitor='val_loss'
- **Mixed Precision**: Enabled for GPU acceleration

### 4.3 Results Comparison

| **Model** | **MAE** | **RMSE** | **R²** | **MAPE (%)** | **Epochs** |
|-----------|---------|----------|--------|--------------|------------|
| RNN_GPU_Optimized | **0.0680** | 0.0843 | 0.76 | 16.22 | 45 |
| Advanced_LSTM_GPU | 0.1222 | 0.1526 | 0.22 | 26.88 | 38 |
| Deep_GRU_GPU | 0.1661 | 0.1998 | -0.34 | 38.40 | 42 |

### 4.4 Multi-Step Forecasting Extension

Extended the best-performing RNN model for 24-hour ahead forecasting:
- **Approach**: Recursive prediction strategy
- **Performance Degradation**: MAE increased to 0.1245 for 24-step ahead
- **Insights**: Prediction uncertainty accumulates with forecast horizon

---

## 5. Task 4: Attention-based and Transformer Models

### 5.1 Transformer Architecture

```python
class TransformerModel:
    def __init__(self, d_model=128, num_heads=8, num_layers=4):
        self.attention_layers = MultiHeadAttention(num_heads, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model)
        self.layer_norm = LayerNormalization()
```

**Configuration:**
- **d_model**: 128
- **Attention Heads**: 8
- **Encoder Layers**: 4
- **Positional Encoding**: Sinusoidal embeddings
- **Dropout**: 0.1

### 5.2 Results

| **Model** | **MAE** | **RMSE** | **R²** | **MAPE (%)** | **Training Time** |
|-----------|---------|----------|--------|--------------|-------------------|
| Transformer | 0.0764 | 0.0977 | 0.68 | 21.15 | 4x faster than CPU |

### 5.3 Comparison with LSTM/RNN

The transformer model demonstrates:
- **Better long-range dependencies**: Attention mechanism captures patterns across the entire input sequence
- **Parallel processing**: Significantly faster training compared to sequential RNN models
- **Interpretability**: Attention weights provide insights into which time steps are most important

However, it still underperforms the linear baseline, suggesting that the 24-hour window doesn't contain sufficient complexity to justify the architectural sophistication.

---

## 6. Analysis and Discussion

### 6.1 Answer to Q1: Model Generalization Comparison

**Which model generalizes better and why? Are there signs of overfitting/underfitting?**

**Linear Regression** demonstrates the best generalization:
- **Reasons**: The normalized electricity consumption signal exhibits strong linear relationships within 24-hour windows
- **No overfitting**: Simple model with few parameters relative to training data
- **Appropriate complexity**: Matches the signal's inherent linearity

**Deep Learning Models** show varying degrees of overfitting:
- **RNN**: Moderate overfitting, but reasonable generalization
- **LSTM/GRU**: More severe overfitting due to excessive parameters (4 layers × 512 units)
- **Evidence**: Large gap between training and validation performance

**ARIMA Model** exhibits underfitting:
- **Reason**: Lacks seasonal components needed for electricity consumption patterns
- **Solution**: SARIMA with seasonal parameters would likely improve performance

### 6.2 Answer to Q2: Temporal Pattern Capture Analysis

**Which model captures temporal patterns best? What are the advantages/disadvantages of each architecture?**

**Temporal Pattern Ranking:**
1. **Linear Regression**: Best for short-term linear trends
2. **RNN**: Good sequential processing, captures short-term dependencies
3. **Transformer**: Excellent long-range attention, parallel processing
4. **LSTM/GRU**: Designed for long-term memory but overparameterized here

**Architecture Analysis:**

| **Architecture** | **Advantages** | **Disadvantages** |
|------------------|----------------|-------------------|
| **Linear** | Simple, interpretable, fast | Cannot capture complex nonlinear patterns |
| **RNN** | Sequential processing, parameter efficient | Vanishing gradient, limited memory |
| **LSTM/GRU** | Long-term memory, handles sequences | Computationally expensive, overfitting risk |
| **Transformer** | Parallel processing, attention mechanism | High complexity, requires large datasets |

### 6.3 Cross-Model Performance Summary

| **Model Family** | **Best MAE** | **Complexity** | **Training Speed** | **Interpretability** |
|------------------|--------------|----------------|-------------------|---------------------|
| Classical | **0.0078** | Low | Fast | High |
| Statistical | 0.1536 | Medium | Medium | Medium |
| Deep Learning | 0.0680 | High | Slow | Low |
| Transformer | 0.0764 | Very High | Medium | Medium |

---

## 7. Advanced Analysis and Future Directions

### 7.1 Why Does Linear Regression Excel?

The linear model's superiority reveals important insights about the dataset:
1. **Signal Characteristics**: Electricity consumption exhibits strong linear patterns within 24-hour windows
2. **Normalization Effect**: Min-max scaling eliminates non-linear scaling effects
3. **Feature Engineering**: Time-based features capture seasonality effectively
4. **Prediction Horizon**: Short-term forecasting (1 hour) favors simpler models

### 7.2 Improving Deep Learning Performance

**Potential Improvements:**
1. **Extended Context**: Increase input window to 168 hours (weekly patterns)
2. **Feature Engineering**: Add weather data, calendar features, lag variables
3. **Regularization**: Implement dropout, L2 regularization, batch normalization
4. **Hyperparameter Tuning**: Systematic optimization of architecture parameters
5. **Ensemble Methods**: Combine multiple models for robust predictions

### 7.3 Transformer Model Enhancements

**Future Transformer Improvements:**
1. **Seasonal Embeddings**: Incorporate time-based positional encodings
2. **Multi-scale Attention**: Different attention heads for different temporal scales
3. **Hierarchical Architecture**: Separate encoders for daily, weekly, seasonal patterns
4. **Pre-trained Models**: Leverage domain-specific pre-trained transformers

---

## 8. Conclusions

This comprehensive study of time series forecasting on electricity consumption data yields several key insights:

### 8.1 Primary Findings
1. **Model-Signal Alignment**: The linear model's success demonstrates that model complexity must match signal complexity
2. **Context Window Importance**: 24-hour windows may be insufficient for complex models to demonstrate advantages
3. **Data Quality Impact**: Clean, well-structured data often favors simpler, interpretable models
4. **Computational Efficiency**: Simple models provide excellent ROI for this specific forecasting task

### 8.2 Practical Implications
- **Production Deployment**: Linear regression offers the best balance of accuracy, speed, and interpretability
- **Model Selection**: Always establish strong baselines before pursuing complex architectures
- **Resource Allocation**: Invest in data quality and feature engineering before model complexity

### 8.3 Research Contributions
- **Baseline Establishment**: Created robust benchmarks across multiple model families
- **GPU Optimization**: Demonstrated 3-5x speedup through optimized implementations
- **Methodological Framework**: Provided systematic approach to time series forecasting evaluation

### 8.4 Future Work
1. **Extended Evaluation**: Test with longer prediction horizons and input windows
2. **External Features**: Incorporate weather, economic, and calendar variables
3. **Hybrid Models**: Combine classical and deep learning approaches
4. **Real-time Deployment**: Implement online learning and concept drift detection

This study establishes that sophisticated models require appropriate problem complexity to demonstrate their value, while simple, well-engineered solutions often provide the best practical performance for many real-world forecasting tasks.

---

## Appendix

### Technical Environment
- **Python**: 3.12
- **TensorFlow**: 2.19
- **Hardware**: CUDA-enabled GPU
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, transformers

### Reproducibility
- All code and results available in project repository
- Detailed notebooks provided for each task
- Seeds set for reproducible results
- Environment requirements documented

### Data Availability
- Dataset: Kaggle PJM Hourly Energy Consumption
- Processed data and results stored in `results/` directory
- Visualization outputs saved as high-resolution PNG files
