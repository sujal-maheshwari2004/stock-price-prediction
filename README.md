# **Stock price Predictor**

This project is a Streamlit-based web application designed to predict stock prices using a Long Short-Term Memory (LSTM) model. The app allows users to visualize stock data, make predictions using a pre-trained model, and even train their own models with custom stock data.

## **Table of Contents**

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Modules Overview](#modules-overview)
- [Future Improvements](#future-improvements)

## **Features**

- **Stock Data Visualization**: View historical stock data and compare it with various moving averages (MA50, MA100, MA200).
- **Stock Price Prediction**: Predict future stock prices using a pre-trained LSTM model.
- **Error Analysis**: Visualize prediction errors and analyze performance using error metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- **Model Training**: Train an LSTM model using custom stock data, save the trained model, and use it for future predictions.

## **Project Structure**

```plaintext
Stock-Market-Predictor/
│
├── main_app.py
├── data_preprocessing.py
├── plotting.py
├── model_prediction.py
├── model_training.py
├── Stock Predictions Model.keras
├── requirements.txt
└── README.md
```

- **`main_app.py`**: The main application script that ties all components together using Streamlit.
- **`data_preprocessing.py`**: Handles data loading, preprocessing, and scaling.
- **`plotting.py`**: Manages plotting of stock prices, moving averages, and prediction errors.
- **`model_prediction.py`**: Contains functions to make predictions with the LSTM model and calculate errors.
- **`model_training.py`**: Contains functions to prepare data, train the LSTM model, and save the trained model.
- **`Stock Predictions Model.keras`**: A pre-trained LSTM model for stock price prediction.
- **`requirements.txt`**: Lists the required Python libraries and packages for the project.
- **`README.md`**: Project documentation.

## **Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sujal-maheshwari2004/stock-price-prediction/
   cd Stock-Market-Predictor
   ```

2. **Set Up a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run main_app.py
   ```

   The app should open in your default web browser.

## **Usage**

### **Predicting Stock Prices**

1. **Launch the Application**: Run `main_app.py` using Streamlit.
2. **Enter Stock Symbol**: Input the stock symbol (e.g., 'GOOG') in the provided text box.
3. **Visualize Stock Data**: View historical data and moving averages.
4. **Predict Prices**: The app will display predicted prices alongside actual prices. Analyze the prediction accuracy using visual plots and error metrics.

### **Training a New Model**

1. **Navigate to the "Train Model" Tab**.
2. **Enter Stock Symbol and Date Range**: Input the stock symbol and select the date range for training data.
3. **Train the Model**: The app will train a new LSTM model based on the provided data and save it for future predictions.

## **Dependencies**

- **numpy**
- **pandas**
- **yfinance**
- **keras**
- **tensorflow**
- **streamlit**
- **matplotlib**
- **scikit-learn**

For detailed version information, see the `requirements.txt` file.

## **Modules Overview**

### **1. `data_preprocessing.py`**
Handles:
- Loading stock data using `yfinance`.
- Scaling and splitting data for training and testing.
  
### **2. `plotting.py`**
Provides:
- Functions to plot stock prices and moving averages.
- Functions to visualize the difference between actual and predicted prices.

### **3. `model_prediction.py`**
Includes:
- Functions to load and use the LSTM model for predictions.
- Error calculation and scaling back predictions to original price range.

### **4. `model_training.py`**
Handles:
- Data preparation for model training.
- Training and saving the LSTM model.

### **5. `main_app.py`**
- Integrates all modules into a cohesive Streamlit application.
- Manages user input, calls necessary functions from other modules, and displays results.

## **Future Improvements**

- **Additional Model Types**: Implement and compare different model architectures (e.g., GRU, Transformer).
- **Real-Time Predictions**: Integrate real-time stock data for up-to-date predictions.
- **Model Hyperparameter Tuning**: Add functionality to tune hyperparameters through the UI.
- **Enhanced Visualization**: Improve the visual appeal and interactivity of plots.
