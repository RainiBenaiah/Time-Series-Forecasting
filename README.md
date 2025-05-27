# Time-Series-Forecasting
# Air Pollution Forecasting Using LSTM Models

## Overview

This project develops a sequence-based time series forecasting model using Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks, to predict hourly PM2.5 concentrations. The goal is to leverage temporal dependencies in historical weather and pollution data to provide accurate air quality forecasts for public health warnings and mitigation planning.

## Problem Statement

Air pollution poses a significant threat to global public health, with PM2.5 (fine particulate matter) being particularly harmful due to its ability to penetrate deep into the lungs. Accurate prediction of PM2.5 levels is critical for:
- Issuing timely public health warnings
- Planning pollution mitigation strategies
- Supporting environmental policy decisions

## Dataset

The dataset consists of hourly air quality and meteorological data split into `train.csv` and `test.csv`.

### Features
- **Numerical Features**: DEWP, TEMP, PRES, Iws, Is, Ir
- **Categorical Features**: Wind Direction (one-hot encoded as cbwd_NW, cbwd_SE, cbwd_cv)  
- **Temporal Features**: datetime (converted to year, month, day, hour, dayofweek)
- **Target Variable**: pm2.5 concentration

### Data Preprocessing

1. **Missing Values**: 1,921 missing values in pm2.5 column handled using:
   - Backward fill followed by random noise injection
   - Ensured non-negativity of PM2.5 readings

2. **Feature Engineering**:
   - Converted datetime into time-based features
   - Removed non-informative columns (No, index, datetime)
   - Applied MinMax scaling to all features and target

3. **Sequence Preparation**: Created 72-hour input sequences with 14 features

## Model Architecture

### Best Performing Model
- **Architecture**: Single-layer LSTM with Dense output
- **LSTM Layer**: 64 units with tanh activation
- **Output Layer**: 1 unit with sigmoid activation
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Regularization**: Dropout (0.3)
- **Batch Size**: 32
- **Input Shape**: (72, 14) - 72-hour sequences with 14 features

### Training Configuration
- **Epochs**: Up to 100 with EarlyStopping (patience=5)
- **Metrics**: Custom RMSE
- **Callbacks**: EarlyStopping to prevent overfitting

## Experimental Results

### Model Performance Summary

| Model Type | Best MSE | Configuration |
|------------|----------|---------------|
| **LSTM** | **0.00357** | 64 units, batch_size=32, lr=0.001, tanh+sigmoid |
| GRU | 0.00412 | 64 units, batch_size=32, lr=0.001, ReLU |

### Key Findings

1. **Activation Functions**: tanh + sigmoid combination outperformed ReLU
   - tanh is smoother and zero-centered, better for time-series
   - sigmoid compresses output values, aligning with normalized targets

2. **Learning Rate**: 0.001 provided optimal convergence
   - Lower rates (0.0001) were stable but slower
   - Higher rates (0.01) caused training instability

3. **Batch Size**: 32 provided the best balance
   - Very small batch sizes (8) caused unstable learning
   - Larger batch sizes didn't significantly improve performance

4. **Regularization**: Dropout (0.3) improved generalization and prevented overfitting

5. **Model Comparison**: LSTM outperformed GRU due to better long-term memory retention

## Installation & Setup

```bash
# Clone the repository
git clone [repository-url]
cd air-pollution-forecasting

# Install required packages
pip install tensorflow pandas numpy scikit-learn matplotlib


## Model Performance Metrics

- **Best MSE**: 0.00357
- **RMSE**: ~0.0598 (on normalized scale)
- **Training Stability**: Achieved through EarlyStopping and dropout regularization

## Future Improvements

1. **Multi-step Forecasting**: Extend to predict next 6 or 24 hours
2. **Advanced Architectures**: 
   - Bidirectional LSTM
   - Stacked LSTM layers
   - Attention mechanisms
   - Transformer-based models
3. **Hyperparameter Optimization**: Implement Bayesian optimization or GridSearchCV
4. **Feature Enhancement**: Include additional meteorological variables
5. **Ensemble Methods**: Combine multiple models for improved accuracy

## Technical Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib/Seaborn (for visualization)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request


## Contact

[b.raini@alustudent.com]

---

**Note**: This project demonstrates the effectiveness of LSTM networks for environmental time series forecasting and provides a foundation for real-world air quality prediction systems.
