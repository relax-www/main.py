import os
import numpy as np
import tensorflow
# import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
# import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score  # 交叉验证法的库
import openpyxl as xl
import pickle

if __name__ == '__main__':
    data=pd.read_csv('data/training_features.csv', encoding="ansi")
    data = pd.read_csv('data/training_features.csv')
    # features = ["Sex", "Age", "SibSp", "Parch", "Fare"]
    #
    # X = pd.get_dummies(train_data[features])
    # X.info()
    # data = pd.read_excel(resource_path, sheet_name="original")
    a=1