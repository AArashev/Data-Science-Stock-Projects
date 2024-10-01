# Enhancing-Stock-Market-Predictions-with-LSTM-Neural-Networks
This project applies an Artificial Recurrent Neural Network called Long Short-Term Memory (LSTM) to predict the closing stock price of Apple Inc. The model is trained on historical stock price data, using the past 60 days of prices to predict future prices. The objective is to demonstrate how LSTM can be used effectively for time-series forecasting in financial data.

## Project Workflow
1. **Data Collection**: Stock price data for Apple Inc. is sourced from Yahoo Finance between 2012 and 2019.
2. **Data Preprocessing**: The data is cleaned and scaled to make it suitable for LSTM modeling.
3. **Model Building**: A Sequential LSTM model is created using `Keras`, with multiple layers to capture the sequential nature of the data.
4. **Model Training**: The model is trained on 80% of the data, using a batch size of 32 and 100 epochs.
5. **Model Evaluation**: The model's performance is evaluated using metrics like RMSE, MAE, and MAPE.
6. **Results Visualization**: The model's predictions are visualized against actual stock prices to assess performance.

## Getting Started

### Prerequisites
- Python 3.8 or above
- Jupyter Notebook or any other IDE for running the notebook
- Libraries: `numpy`, `pandas`, `pandas_datareader`, `yfinance`, `scikit-learn`, `keras`, `matplotlib`

To install the necessary libraries, run:
```bash
pip install numpy pandas pandas-datareader yfinance scikit-learn keras matplotlib




Project Details
Data Collection

    The stock data for Apple (AAPL) is fetched using yfinance, covering the period from January 1, 2012, to December 17, 2019.

Model Description

    LSTM Architecture: The LSTM model is constructed using Keras. The model includes:
        2 LSTM layers with 50 neurons each
        Dropout layers to prevent overfitting
        Dense layers for the output
    Training: The model is trained on 80% of the data with 60 time steps (look-back period).

Model Evaluation

The model is evaluated using several metrics:

    Root Mean Squared Error (RMSE): Measures the average magnitude of the prediction errors.
    Mean Absolute Error (MAE): Measures the average absolute errors between predicted and actual values.
    Mean Absolute Percentage Error (MAPE): Measures the accuracy as a percentage.

Results Visualization

The actual stock prices and the model's predictions are plotted using matplotlib to visualize how well the model tracks the historical data.
Results and Discussion

The model demonstrates the ability to predict stock price trends reasonably well. The performance is validated using a time-series cross-validation with 5 splits, showing different RMSE, MAE, and MAPE values across folds. Below are the metrics for each fold:
Fold	RMSE	MAE	MAPE (%)
1	1.68	1.21	5.55
2	5.99	5.84	19.35
3	1.15	0.89	3.20
4	1.89	1.53	3.47
5	2.43	2.11	4.23

The model's predictions are also visualized against the actual prices, providing a visual assessment of its accuracy.
Future Work

    Parameter Tuning: Optimize model hyperparameters such as the number of neurons, dropout rate, and learning rate to improve accuracy.
    Experiment with Other Architectures: Compare the LSTM model with other recurrent architectures like GRUs or attention-based models.
    Incorporate Additional Features: Use other stock-related features such as trading volume, indicators, or macroeconomic data.

Acknowledgments

    Data sourced from Yahoo Finance.
    Libraries and tools: yfinance, pandas, scikit-learn, Keras, matplotlib.

License

This project is licensed under the MIT License - see the LICENSE.md file for details.
