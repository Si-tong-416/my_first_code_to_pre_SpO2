# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
# load data
dataset = loadtxt('train.csv', delimiter=",")
# # split data into X and y
# X_train = dataset[:,0:8]
# y_train = dataset[:,8]

# split data into X and y
X_train = dataset[:,1:]  # 从第二列开始到最后一列作为特征
y_train = dataset[:,0]   # 第一列作为标签


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open("pima_model_me.pkl", "wb"))
