# Realized-Volatility-Prediction

## 1)-Data Engineering

### 1.1) Constructing the Features
We used the data from [this](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data) Kaggle competition. 

First, the data is cleaned; then we use [this](https://github.com/alins95/Realized-Volatility-Prediction/blob/main/code/feature_eng_notebook.ipynb) notebook to generate our features based on the trade and order book data. We compute the features for the whole 10 minuet window and the last 100,200, and 300 seconds of each window. The results are saved in df_train.csv.

### 1.2) Reconstructing the Time Order
We used the df_train and scipy.isomap to reconstruct the order of time_id's in the data set. We do not use the time as an input to our models; instead, we only used this order to split the data based on time. This helps us to avoid future looking in our models, and gives better genarilzation when we run the model on real time data. The notebook used for reconstruction of time order is [here](https://github.com/alins95/Realized-Volatility-Prediction/blob/main/code/recovering_time.ipynb).

## 2)-Neural Network Model

The NN architecture has eight hidden layers. The model has an embedding layer with(embedding dimension = 50) that keeps track of different stock_id's. In the forward method, we apply a sigmoid to the ouput of our network, since volatality is positive, and all the data points have volatality less than 1. There are no batch normalization or drop out layers, since the activation function in our model is SELU which gives us normalization effect during training. We use Lecun method to initialize our model; this leads to a faster training loop because of the SELU activation. The notebook used for training the NN model can be found [here](https://github.com/alins95/Realized-Volatility-Prediction/blob/main/code/Neural_Net_Model.ipynb).

## 3)- LGBM Model

## 4)- Ensemble Model
We used a simple convex combination of the two models as our ensemble. The resulting model is defined as follows:

Ensemble_model = alpha * (NN_model) + (1-alpha) * (LGBM_model).

We optimize the validation loss to find the best alpah. The NN_model has 0.2377 validation loss, and the LGBM_model has 0.2393 loss while the Ensemble one has 0.2325 validation loss. The ensemble notebook can be found in [this](https://github.com/alins95/Realized-Volatility-Prediction/blob/main/code/Ensemble.ipynb) link.
 
