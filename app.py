import streamlit as st
from data_preprocessing import load_and_preprocess_data
from plotting import plot_moving_averages, plot_original_vs_predicted, plot_errors
from model_prediction import make_predictions, scale_predictions, calculate_error_metrics
from model_training import prepare_data, create_and_train_model, save_model_to_file

# Streamlit UI
st.title('Stock Market Predictor and Model Trainer')

tab1, tab2 = st.tabs(["Predict Stock Prices", "Train Model"])

# Tab 1: Predict Stock Prices
with tab1:
    st.header('Stock Price Prediction')
    
    stock = st.text_input('Enter Stock Symbol', 'GOOG')
    start = st.date_input('Start Date', value=pd.to_datetime('2012-01-01'))
    end = st.date_input('End Date', value=pd.to_datetime('2022-12-31'))
    model_path = 'Stock Predictions Model.keras'

    if st.button('Predict'):
        # Load and preprocess data
        data, data_train, data_test, scaler, data_test_scaled = load_and_preprocess_data(stock, start, end)

        st.subheader('Stock Data')
        st.write(data)

        # Plot moving averages
        fig1, fig2, fig3 = plot_moving_averages(data)
        st.subheader('Price vs MA50')
        st.pyplot(fig1)
        st.subheader('Price vs MA50 vs MA100')
        st.pyplot(fig2)
        st.subheader('Price vs MA100 vs MA200')
        st.pyplot(fig3)

        # Make predictions
        x, y, predict = make_predictions(model_path, data_test_scaled)
        predict, y = scale_predictions(predict, y, scaler)

        # Plot predictions
        fig4 = plot_original_vs_predicted(y, predict)
        st.subheader('Original Price vs Predicted Price')
        st.pyplot(fig4)

        # Plot errors
        errors = y - predict
        fig5 = plot_errors(errors)
        st.subheader('Prediction Errors Over Time')
        st.pyplot(fig5)

        # Calculate and display error metrics
        mse, mae = calculate_error_metrics(y, predict)
        st.subheader('Error Metrics')
        st.write(f'Mean Squared Error: {mse}')
        st.write(f'Mean Absolute Error: {mae}')

# Tab 2: Train Model
with tab2:
    st.header('Train Stock Prediction Model')
    
    stock_train = st.text_input('Enter Stock Symbol for Training', 'GOOG')
    start_train = st.date_input('Training Start Date', value=pd.to_datetime('2012-01-01'))
    end_train = st.date_input('Training End Date', value=pd.to_datetime('2022-12-31'))
    model_path_train = 'Stock Predictions Model.keras'

    if st.button('Train Model'):
        # Prepare data
        _, _, _, _, data_test_scaled = prepare_data(stock_train, start_train, end_train)

        # Create and train the model
        model = create_and_train_model(data_test_scaled)

        # Save the trained model
        save_model_to_file(model, model_path_train)

        st.write(f"Model trained and saved as '{model_path_train}'.")
