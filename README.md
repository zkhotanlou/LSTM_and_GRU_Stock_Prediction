# Stock Price Prediction Project

**Stock Price Prediction** is a project that uses deep learning models to predict the stock prices of a given company based on historical data. 

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
4. [Data](#data)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Architectures](#model-architectures)
7. [Parameter Tuning](#parameter-tuning)
8. [Evaluation](#evaluation)
9. [Comparing Models](#comparing-models)

## Introduction

Stock price prediction is a challenging task due to the complex and dynamic nature of financial markets. Factors like historical price data, trading volumes, market sentiment, and external events all play a significant role in determining the future trajectory of stock prices. The incorporation of these diverse features into predictive models requires advanced machine learning and deep learning methods.
In this project, we leverage the power of Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), to capture the temporal dependencies in historical stock price data. These models are capable of learning complex patterns and relationships in time series data, making them well-suited for stock price forecasting.

## Getting Started

Before you start, make sure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Necessary Python libraries like NumPy, Pandas, Matplotlib, Keras, TensorFlow, and scikit-learn.

## Data
Google stock data from 2006 to early 2014, where data from the first day each month was collected. A data frame with 98 observations on the following 7 variables:
date, open, high, low, close, volume, adj_close

## Data Preprocessing

We isolate the `close` column and normalize its values between 0 and 1 for uniformity in model training. Data preprocessing also involves segmenting the data into input-output pairs. We use a window size (controlled by the `step_size`) and consider each window of data as one training example. For each window, the last day is considered as the output label. We then divide the data into training and testing sets, reshaping the inputs to meet the requirements of LSTM and GRU networks.

## Model Architectures

We've built and trained several deep learning models with different architectures to find the best approach for stock price prediction. Let's explore them:
### LSTM Models

1. **Single-Layer LSTM:** This model consists of a single LSTM layer with 60 units, a dropout layer to prevent overfitting, and a dense output layer. It uses Mean Squared Error (MSE) as the loss function, ADAM as the optimizer, and tanh as the activation function. This model's structure is as follows:  
![Single-laye LSTM Model Structure](https://github.com/zkhotanlou/LSTM_and_GRU_Stock_Prediction/assets/84021970/42e0176d-d240-46e1-b13c-bed32a60ceb1)

3. **Two-Layer LSTM:** Similar to the previous model, but with two stacked LSTM layers. It has 43,981 learnable parameters.  
![Two-layer LSTM Model Structure](https://github.com/zkhotanlou/LSTM_and_GRU_Stock_Prediction/assets/84021970/346e948e-f9d7-47e2-9288-148375202473)

4. **Three-Layer LSTM:** A more complex LSTM model with three stacked LSTM layers, resulting in 50,851 learnable parameters.  
![Three-layer LSTM Model Structure](https://github.com/zkhotanlou/LSTM_and_GRU_Stock_Prediction/assets/84021970/e83658fa-062b-4e32-b65f-293c28f4710a)

### GRU Models

1. **Single-Layer GRU:** This GRU model includes one GRU layer with 50 units, a dense layer with 512 neurons, and a final dense layer with one neuron in the output. We use a dropout layer with a rate of 0.2 to prevent overfitting. The model is trained and loss trends are plotted. The architecture is as follows:  
![Single-Layer GRU](https://github.com/zkhotanlou/LSTM_and_GRU_Stock_Prediction/assets/84021970/17808066-42be-47e6-93c5-30b73cc2b63e)

2. **Two-Layer GRU:** Similar to the previous model but with two stacked GRU layers.  
![Two-Layer GRU](https://github.com/zkhotanlou/LSTM_and_GRU_Stock_Prediction/assets/84021970/51c2efff-0fe3-435b-ab83-5c2aab6f631c)

3. **Three-Layer GRU:** A more complex GRU model with three stacked GRU layers.  
![Three-Layer GRU](https://github.com/zkhotanlou/LSTM_and_GRU_Stock_Prediction/assets/84021970/a058f34d-4b05-4d70-89fe-af75648c6a6f)

## Parameter Tuning

To fine-tune the models, we have used various parameter settings. For instance, we iterated through different numbers of units in the RNN layers, running multiple experiments to choose the configuration that minimized the average error. 
![value of RSME in several number of units - LSTM model](https://github.com/zkhotanlou/LSTM_and_GRU_Stock_Prediction/assets/84021970/f324f155-598b-4f51-8c34-81fe2e557472)  
value of RSME in several number of units - LSTM model

![value of RSME in several number of units - GRU model](https://github.com/zkhotanlou/LSTM_and_GRU_Stock_Prediction/assets/84021970/9623ab98-96cf-4e07-983f-c3aa1e3561fc)  
value of RSME in several number of units - GRU model

## Evaluation

For each model, we trained the network and assessed its performance:
- We used Root Mean Squared Error (RMSE) for the predicted and actual data, both for normalized and original data.
- We also calculated the Mean Absolute Percentage Error (MAPE) for the test data.
- Loss trend graphs over epochs provide insights into the training process.

The figures below shows the best model's results:  

![value of loss during the train process](https://github.com/zkhotanlou/LSTM_and_GRU_Stock_Prediction/assets/84021970/7b4c0504-4549-43de-b6d6-98248df030a9)  
value of loss during the train process  


![Close price prediction](https://github.com/zkhotanlou/LSTM_and_GRU_Stock_Prediction/assets/84021970/51df87aa-2b34-4492-be4a-1b515910dfed)  
Close price prediction

## Comparing Models
| Model     | Mean RMSE | Best RMSE |
| --------- | --------- | --------- | 
| **LSTM1** | **19.81** | **6.04**  |
| LSTM2     | 64.20     | 26.21     |
| LSTM3     | 64.20     | 26.21     |
| GRU1      | 25.10     | 9.14      |
| GRU2      | 38.48     | 6.09      |
| GRU3      | 67.80     | 31.25     |
