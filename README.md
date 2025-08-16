# Realized-Volatility-Prediction

## 1) Data Engineering

### 1.1) Constructing the Features
We used  data from [this](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data) Kaggle competition. 

In the [feature engineering notebook](https://github.com/alins95/Realized-Volatility-Prediction/blob/main/code/1.1_feature_engineering.ipynb), we first clean the data and generate our features based on the trade and order book data. We compute the features for the whole 10 minute window and the last 100,200, and 300 seconds of each window. The results are saved in df_train.csv.

### 1.2) Reconstructing the Time Order
Using the engineered features and scipy.isomap, we try to reconstruct the order of time_id's in the data set in the [recovering_time notebook](https://github.com/alins95/Realized-Volatility-Prediction/blob/main/code/1.2_recovering_time.ipynb) notebook. We do not use the time as an input to our models; instead, we only used this order to split the data based on time, i.e. using teh earlier data points for training and the remaining data points for validation. This helps us avoid future looking in our models, and ensures better genarilzation when we run the model on real time data.

## 2) Neural Network Model

In the [neural_net_model notebook](https://github.com/alins95/Realized-Volatility-Prediction/blob/main/code/2_neural_net_model.ipynb), we train a NN model for predicting the volatilities. The NN architecture has eight hidden layers. The model has an embedding layer with(embedding dimension = 50) that keeps track of different stock_id's. In the forward method, we apply a sigmoid to the ouput of our network, since volatality is positive, and all the data points have volatality less than 1. There are no batch normalization or drop out layers, since the activation function in our model is SELU which gives us normalization effect during training. We use Lecun method to initialize our model; this leads to a faster training loop because of the SELU activation. 

## 3) LightGBM Model

Next, in the [lgbm_model notebook](https://github.com/alins95/Realized-Volatility-Prediction/blob/main/code/3_lgbm_model.ipynb), we train and optimize a boosted tree using optuna. Similar to the NN model, we use RMSPE loss function. 

## 4) Ensemble Model
For the final prediction, in the [ensemble notebook](https://github.com/alins95/Realized-Volatility-Prediction/blob/main/code/4_ensemble.ipynb), we used a simple convex combination of the two models as our ensemble. The resulting model is defined as follows:

Ensemble_model = alpha * (NN_model) + (1-alpha) * (LGBM_model).

We optimize the validation loss to find the best alpah. The NN_model has 0.2377 validation loss, and the LGBM_model has 0.2393 loss while the Ensemble one has 0.2325 validation loss.

 
