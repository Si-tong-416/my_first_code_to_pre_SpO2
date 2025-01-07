import pickle
import re

import joblib
import numpy as np
import os
import pandas as pd
from numpy import mean
from pygam import LinearGAM
from scipy.stats import uniform
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR, LinearSVR, LinearSVC, SVC
from SPO2_Pred import CalSpo22
from pickle import load
import shutil
from sklearnex import patch_sklearn, unpatch_sklearn
import os

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error, max_error, median_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import numpy as np
from numpy import std, mean, sqrt
from sklearn.metrics import confusion_matrix
from statistics import mean, median
from sklearn.model_selection import KFold
import argparse
import math
import cv2
from scipy.ndimage import zoom
import csv

if __name__ == '__main__':

    # folder_path = "F:/UBFC-rPPG-original"
    # folder_path = "F:/123"
    # #
    # with open(f"C:/Users/YF/Desktop/subject/30000/SVR.csv", 'a',
    #           newline=''
    #           ) as csvfile:
    #     writer1 = csv.writer(csvfile)
    #     writer1.writerow(
    #         ["subject", 'MAE', "MSE", "MAPE", "Pearson"
    #             # , "best_param"
    #          ]
    #     )
    # with open(f"C:/Users/YF/Desktop/subject/123/all.csv", 'a',
    #           newline=''
    #           ) as csvfile:
    #     writer1 = csv.writer(csvfile)
    #     writer1.writerow(
    #         ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10', 'var11', 'var12', 'var13', 'var14', 'var15', 'var16', "var17", "var18", 'ppg',
    #          'spo2'
    #          ]
    #     )
    # for i in range(42):
    #     with open(f"C:/Users/YF/Desktop/subject/subject{i+1}.csv", 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(
    #             ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10', 'var11', 'var12','ppg',
    #              'spo2']
    #             )

    # CalSpo22('F:/UBFC-rPPG-original/subject1/vid.avi', 'F:/UBFC-rPPG-original/subject1/ground_truth.txt')


    # 遍历文件夹中的子文件
    # for sub_folder in os.listdir(folder_path):
    #     sub_folder_path = os.path.join(folder_path, sub_folder)
    #
    #     # 判断子文件是否为文件夹
    #     if os.path.isdir(sub_folder_path):
    #         # 读取视频文件
    #         video_file = os.path.join(sub_folder_path, "vid.avi")
    #         # 进行视频文件的处理操作，例如读取、处理等
    #         # 读取txt文件
    #         bvp_file = os.path.join(sub_folder_path, "ground_truth.txt")
    #         print(video_file)
    #         print(bvp_file)
    #
    #         a, b = CalSpo22(video_file
    #                         # , bvp_file
    #                         # , sub_folder
    #                         )

    #
    #         data1 = pd.read_csv(f'C:/Users/YF/Desktop/subject/123/{sub_folder}_L_R_F.csv')
    #         predictors = [
    #             'var1', 'var2', 'var3', 'var4', 'var5', 'var6',
    #             'var7', 'var8', 'var9', 'var10', 'var11', 'var12',
    #             'var13', 'var14', 'var15', 'var16', 'var17', 'var18',
    #             "ppg"
    #                       ]
    #         if data1['ppg'].isna().any():
    #             data1['ppg'] = b
    #         else:
    #             pass
    #
    #         x_1 = data1[np.array(predictors)].values
    #         outcome = ['spo2']
    #         c = data1[outcome].isna().any()
    #         print(c)
    #
    #         # x_2 = data2[np.array(predictors)].values
    #         if c.bool():
    #             y_1 = a
    #
    #         else:
    #             y_1 = data1[outcome]
    #
    #
    #         # y_1 = data1[outcome]
    #         x_tran, x_test, y_train, y_test = train_test_split(x_1, y_1, test_size=0.25, random_state=42)
    #         LG = SGDRegressor()
    #         # LG = LinearRegression()
    #         # LG = SVR(kernel='linear', gamma='scale', C=55)
    #         # par = {
    #         #     'kernel': ['linear'],
    #         #     'C': np.arange(1, 100, 0.1),
    #         #     'gamma': ['auto', 'scale'],
    #         # }
    #         # LG = RandomizedSearchCV(re, param_distributions=par, n_iter=10)
    #         LG.fit(x_tran, y_train)
    #
    #         # print("Best Par", LG.best_params_)
    #         y_hat = LG.predict(x_test)
    #
    #         print("spo2:", mean_absolute_error(y_test, y_hat))
    #         print("spo2:", mean_squared_error(y_test, y_hat))
    #         print("spo2:", mean_absolute_percentage_error(y_test, y_hat))
    #         print("spo2:", r2_score(y_test, y_hat))
    #
    #         with open(f"C:/Users/YF/Desktop/subject/svc/data_svc_ppg_L_R_F.csv", 'a',
    #                   newline=''
    #                   ) as csvfile:
    #             writer = csv.writer(csvfile)
    #             writer.writerow(
    #                 [{sub_folder}, mean_absolute_error(y_test, y_hat), mean_squared_error(y_test, y_hat),mean_absolute_percentage_error(y_test, y_hat), r2_score(y_test, y_hat)
    #                  # , LG.best_params_
    #                  ]
    #             )

            # r = len(x_test) + 1
            # print(y_test)
            # print(y_hat)

            # plt.plot(np.arange(1, 101), y_hat[100: 200], 'go-', label="predict")
            # plt.plot(np.arange(1, 101), y_test[100: 200], 'co-', label="real")
            # plt.legend()
            # plt.show()


    # folder_path = "F:/456"
    #
    # with open("C:/Users/YF/Desktop/12.csv", 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(
    #         ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10', 'var11', 'var12', 'ppg',
    #          'spo2'])
    #
    # 遍历文件夹中的子文件
    # for sub_folder in os.listdir(folder_path):
    #     sub_folder_path = os.path.join(folder_path, sub_folder)
    #
    #     # 判断子文件是否为文件夹
    #     if os.path.isdir(sub_folder_path):
    #         # 读取视频文件
    #         video_file = os.path.join(sub_folder_path, "vid.avi")
    #         # 进行视频文件的处理操作，例如读取、处理等
    #         # 读取txt文件
    #         bvp_file = os.path.join(sub_folder_path, "ground_truth.txt")
    #         print(video_file)
    #         print(bvp_file)
    #         CalSpo22(video_file, bvp_file)



    # video_file = "F:/UBFC-rPPG-original/subject1/vid.avi"
    # bvp_file = "F:/UBFC-rPPG-original/subject1/ground_truth.txt"
    # CalSpo2(video_file, bvp_file)
    # np.random.seed(0)

    # patch_sklearn()

    # data1 = pd.read_csv(f'C:/Users/YF/Desktop/subject/30000/88_108.csv')
    # # data2 = pd.read_csv(f'C:/Users/YF/Desktop/subject/test/12345.csv')
    # # #
    # predictors = [
    #         'var1', 'var2', 'var3', 'var4', 'var5', 'var6',
    #         'var7', 'var8', 'var9', 'var10', 'var11', 'var12',
    #         'var13', 'var14', 'var15', 'var16', 'var17', 'var18',
    #         # "ppg"
    #               ]
    # # #
    # # #
    # outcome = ['spo2']
    # x_2 = data1[np.array(predictors)].values
    # y_2 = data1[outcome]
    #
    # model = joblib.load('ExtraTreesRegressor.pkl')
    #
    # y_hat = model.predict(x_2)
    #
    # print(y_hat)
    # print("Spo2:", np.mean(y_hat))
    #
    # r = len(x_2) + 1
    # plt.plot(np.arange(1, r), y_hat, 'go-', label="predict")
    #
    # plt.legend()
    # plt.show()

    # x_train, x_test, y_train, y_test = train_test_split(x_2, y_2, test_size=0.25)
    # x_train, x_test, y_train, y_test = x_1, x_2, y_1, y_2
    # x_train, y_train = x_2, y_2
    # LG = LinearGAM()
    # par = {
    #     'kernel': ['linear'],
    #     'n_splines': np.arange(1, 100, 1),
    #     'gamma': ['auto', 'scale'],
    # }
    # re = RandomizedSearchCV(LG, param_distributions=par, n_iter=10,random_state=42)
    # lams = np.random.rand(100, n_features)
    # lams = lams * n_features - 3
    # lams = np.exp(lams)

    # clf = LinearRegression()
    # clf1 = LinearSVR()
    # clf = LinearSVC()
    # clf = LinearGAM()
    # clf1 = ensemble.GradientBoostingClassifier()
    # clf1 = ensemble.ExtraTreesClassifier()
    # clf1 = ensemble.RandomForestRegressor()
    # clf = ensemble.AdaBoostRegressor()
    # clf = ensemble.BaggingRegressor()
    # clf = ensemble.HistGradientBoostingRegressor()
    # model = ensemble.BaggingRegressor()
    # model.fit(x_train, y_train)
    # pickle.dump(model, open("BaggingRegressor.pkl", "wb"))
    # clf1.fit(x_train, y_train)
    # y_hat = clf1.predict(x_test)
    # for train_index, test_index in kf.split(X, y):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     # x_train, x_test, y_train, y_test = train_test_split(x_2, y_2, test_size=0.4, random_state=42)
    #     clf = SVR(kernel='linear')
    #     clf.fit(X_train, y_train)
    #     y_hat = clf.predict(X_test)
    #
    #     print("spo2:", mean_absolute_error(y_test, y_hat))
    #     print("spo2:", mean_squared_error(y_test, y_hat))
    #     print("spo2:", mean_absolute_percentage_error(y_test, y_hat))
    #     print("spo2:", r2_score(y_test, y_hat))
    #     print('-------------------')
    #
    #     with open(f"C:/Users/YF/Desktop/subject/30000/SVR.csv", 'a',
    #               newline=''
    #               ) as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(
    #             ['SVR', mean_absolute_error(y_test, y_hat), mean_squared_error(y_test, y_hat),mean_absolute_percentage_error(y_test, y_hat), r2_score(y_test, y_hat)
    #              # , LG.best_params_
    #              ]
    #         )
    #
    #     n = 90
    #     # r = len(x_test) + 1
    #     y_hat_mean = [np.mean(y_hat[i:i + n]) for i in range(0, len(y_hat), n)]
    #     y_test_mean = [np.mean(y_test[i:i + n]) for i in range(0, len(y_test), n)]
    #     r = len(y_test_mean) + 1
    #     plt.plot(np.arange(1, r), y_hat_mean, color='deepskyblue', label="Predict")
    #     plt.plot(np.arange(1, r), y_test_mean, color='orangered', label="True")
    #     plt.legend()
    #     plt.savefig('C:/Users/YF/Desktop/subject/fig/SVR.png')
    #     plt.show()

    # lr = SVR(kernel="linear", C=50)
    # lr.fit(y_hat.reshape(-1, 1), y_test)
    # final_y_hat = lr.predict(y_hat.reshape(-1, 1))
    # print("spo2:", r2_score(y_test, final_y_hat))
    # print(y_test)


    # model = RandomForestRegressor(n_estimators=100, random_state=0)
    # model.fit(x_tran, y_train)
    # y_hat = model.predict(x_test)
    # print("spo2:", r2_score(y_test, y_hat))
    # print(y_test)
    # plt.plot(np.arange(1, r), y_hat, 'go-', label="predict")
    # plt.plot(np.arange(1, r), y_test, 'co-', label="real")
    # plt.legend()
    # plt.show()

    # svr = SVR()
    # par = {
    #     'kernel': ['linear'],
    #     'C': np.arange(0.1, 100, 0.1),
    #     'gamma': ['auto', 'scale'],
    # }
    # re = RandomizedSearchCV(svr, param_distributions=par, n_iter=10, cv=kf, random_state=42)
    # re.fit(x_tran, y_train)
    # print("Best Par", re.best_params_)
    # y_pred = re.predict(x_test)
    # y_hat = model.predict(x_test)

    # print("spo2:", mean_absolute_error(y_test, y_hat))
    # print("spo2:", mean_squared_error(y_test, y_hat))
    # print("spo2:", mean_absolute_percentage_error(y_test, y_hat))
    # print("spo2:", r2_score(y_test, y_hat))

    # # r = len(x_test) + 1
    # # print(y_test)
    # # print(y_hat)

    # plt.plot(np.arange(1, 101), y_hat[100: 200], 'go-', label="predict")
    # plt.plot(np.arange(1, 101), y_test[100: 200], 'co-', label="real")
    # plt.legend()
    # plt.show()

    with open(f"C:/Users/YF/Desktop/subject/DDMPFV/sum.csv", 'a',
              newline=''
              ) as csvfile:
        writer1 = csv.writer(csvfile)
        writer1.writerow(
            ['True', "PRE"]
        )

    folder_path = "C:/Users/YF/Desktop/采集"
    true = []
    pre = []
    # 遍历文件夹中的子文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 判断子文件是否为文件夹
            if file.endswith(".avi"):
                video_path = os.path.join(root, file)
                CalSpo22(video_path, video_path)

                path = video_path.replace("\\", "")
                path1 = path.replace(".avi", "")
                path2 = path1.replace("C:/Users/YF/Desktop/", "")
                data1 = pd.read_csv(f'C:/Users/YF/Desktop/subject/DDMPFV/{path2}.csv')
                predictors = [
                    'var1', 'var2', 'var3', 'var4', 'var5', 'var6',
                    'var7', 'var8', 'var9', 'var10', 'var11', 'var12',
                    'var13', 'var14', 'var15', 'var16', 'var17', 'var18',
                    # "ppg"
                ]

                outcome = ['spo2']
                x_1 = data1[np.array(predictors)].values
                y_1 = data1[outcome]
                model = joblib.load('HistGradientBoostingRegressor.pkl')
                y_hat = model.predict(x_1)
                # print(y_hat)
                print("pre_Spo2:", np.mean(y_hat))
                p = np.mean(y_hat)
                pre.append(p)

            if file == '记录.xlsx':
                excel_path = os.path.join(root, file)
                df = pd.read_excel(excel_path)
                a = df.iloc[2, 2]
                b = df.iloc[3, 2]
                video_path = os.path.join(root, file)
                print("true_Spo2:", a)
                print("true_Spo2:", b)
                true.append(a)
                true.append(b)
                print(video_path)

    data = pd.read_csv('C:/Users/YF/Desktop/subject/DDMPFV/sum.csv')
    data['True'] = true
    data.to_csv('C:/Users/YF/Desktop/subject/DDMPFV/sum.csv', index=False)

    data = pd.read_csv('C:/Users/YF/Desktop/subject/DDMPFV/sum.csv')
    data['PRE'] = pre
    data.to_csv('C:/Users/YF/Desktop/subject/DDMPFV/sum.csv', index=False)

    print(true)
    print(pre)




