# coding:UTF-8
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import pylab as pl
import ReadData
import LR
import BP
import CART
import Result
import Utils
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
warnings.filterwarnings('ignore')

def remove_outliers(df, column , minimum, maximum):
    #去除离群点
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values<minimum, col_values>maximum), col_values.mean(), col_values)
    return df

col_names =  ["Recency", "Frequency", "Monetary", "Time","label"]
fileName="blood/transfusion.data"
path = os.path.realpath(os.curdir)
fileNameFullPath = os.path.join(path, fileName)

def TestBloodDatasetWithLRAkgorithm(path):
    TrainData, TrainLabel, testData,testLabel = ReadData.LoadBloodData(path)
    ProcessedTrainData = Utils.Normalization(TrainData)
    ProcessedTestData = Utils.Normalization(testData)
    w = LR.TrainModel(ProcessedTrainData, TrainLabel, 10000, 0.8, 10001)
    print("Trained Weight:\t")
    print(w)

    # Predit the train data
    PredictResult = LR.PredictWithModel(ProcessedTrainData, w)
    Result.DrawConfusionMatrix(TrainLabel,PredictResult)

    # Predict the test data
    PredictTestResult = LR.PredictWithModel(ProcessedTestData, w)
    Result.DrawConfusionMatrix(testLabel,PredictTestResult)

def LoadBloodData():
    blood = pd.read_csv(fileNameFullPath, header=1, names=col_names)
    X1 = blood.drop('label',axis = 1).values

    #去除立群点
    min_val = blood["Recency"].min()
    max_val = 50
    X = remove_outliers(df=blood, column='Recency' , minimum=min_val, maximum=max_val)
    beta = np.mat(np.ones((blood.shape[0], 1)))
    # 加上B 列
    X = np.append(X1,beta, 1)
    y = blood['label'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    Y_train = np.mat(Y_train).T
    Y_test =np.mat(Y_test).T
    return  X_train, X_test, Y_train,Y_test

def TestBloodLRCase(X_train, X_test, Y_train,Y_test):
    w = LR.TrainModel(X_train, Y_train, 10000, 0.8, 100001)
    PredictResult = LR.PredictWithModel(X_train, w)

    Result.PrintResult(Y_train, PredictResult)
    #Result.DrawConfusionMatrix(np.mat(Y_train).T,PredictResult)

    PredictResult2 = LR.PredictWithModel(X_test, w)
    #Result.DrawConfusionMatrix(np.mat(Y_test).T,PredictResult)
    print("Accuracy :\t ", accuracy_score(Y_test, PredictResult2))
    Result.PrintResult(Y_test, PredictResult2)

def TestBloodBPCase(X_train, X_test, Y_train,Y_test):
    m = np.shape(Y_train)[0]
    lTrainLabel = np.mat(np.zeros((m, 2)))
    for i in range(m):
        lTrainLabel[i, Y_train[i]] = 1
    StandardTrainData = normalize(X_train, axis=0, norm='max')
    w0, w1, b0, b1 = BP.TrainBPModel(StandardTrainData, lTrainLabel, 5, 10000, 0.005, 2, 1000)
    result = BP.PredictWithModel(StandardTrainData, w0, w1, b0, b1)
    BP.PersistModel(w0, w1, b0, b1)
    Result.PrintResult(Y_train, np.mat(np.argmax(result,axis = 1)))
    n = np.shape(Y_train)[0]
    lTestLabel = np.mat(np.zeros((n, 2)))

    StandardTestData = normalize(X_test, axis=0, norm='max')
    result2 = BP.PredictWithModel(StandardTestData, w0, w1, b0, b1)
    Result.PrintResult(Y_test,np.mat(np.argmax(result2,axis = 1)))

def TestBloodCARTCase():
    blood = pd.read_csv(fileNameFullPath, header=1, names=col_names)
    Y = blood['label'].values
    X = blood.values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    Y = np.mat(Y_train)
    
    labels = ["Recency", "Frequency", "Monetary", "Time"]
    lLabelsTmp = labels[:]
    X_Y_Train = X_train.tolist()
    desicionTree = CART.TrainCART(X_Y_Train, lLabelsTmp)
    #CART.treePlotter.createPlot(desicionTree)
    
    # Classfiy need to delete the label column
    t2 = np.delete(X_train, 4 ,axis = 1)
    t2 = t2.tolist()
    lLabelsTmp2 = ["Recency", "Frequency", "Monetary", "Time"]
    result = CART.ClassifyAll(desicionTree, lLabelsTmp2, t2)
    Result.PrintResult(Y_train, result)
    testData = np.delete(X_test, 4 ,axis = 1)
    testData = testData.tolist()
    lLabelsTmp3 = ["Recency", "Frequency", "Monetary", "Time"]
    result = CART.ClassifyAll(desicionTree, lLabelsTmp3, testData)
    Result.PrintResult(Y_test, result)

if __name__ == "__main__":
    X_train, X_test, Y_train,Y_test = LoadBloodData()
    #TestBloodLRCase(X_train, X_test, Y_train,Y_test)
    #TestBloodBPCase(X_train, X_test, Y_train,Y_test)
    TestBloodCARTCase()
