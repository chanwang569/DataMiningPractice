import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os 
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from  sklearn import tree
from sklearn.tree import export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
import Result

column_names= ["label","lepton1pt","lepton1eta","lepton1phi",\
    "lepton2pt","lepton2eta","lepton2phi","missingEnergyMagnitude",\
    "missingEnergyPhi", "MET_rel", "axialMET","M_R",\
    "M_TR_2", "R", "MT2", "S_R","M_Delta_R",
    "dPhi_r_b", "cos(theta_r1)"]


def ReadSUSYData():
    fileName="SUSY/SUSY.csv"
    path = os.path.realpath(os.curdir)
    fileNameFullPath = os.path.join(path, fileName)
    df = pd.read_csv(fileNameFullPath, header=None,names=column_names)
    X = df.drop('label',axis = 1).values
    Y = df["label"].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.fit_transform(X_train)

    pca = PCA(n_components=10)
    pca.fit(X_train)
    X_train = pca.fit_transform(X_train)

    scaler.fit(X_test)
    X_test = scaler.fit_transform(X_test)
    pca.fit(X_test)
    X_test = pca.fit_transform(X_test)
    return X_train, X_test, Y_train, Y_test

def TrainAndTest(algotithm, X_train, X_test,Y_train, Y_test ):
    TrainStarttime = datetime.datetime.now()
    algotithm.fit(X_train, Y_train)
    TrainEndTime = datetime.datetime.now()
    print("Train Time(seconds): ",(TrainEndTime - TrainStarttime).seconds)
    
    PredictStarttime = datetime.datetime.now()
    predictions = algotithm.predict(X_test)
    PredictEndTime = datetime.datetime.now()
    print("Predict Time(seconds): ",(PredictEndTime - PredictStarttime).seconds)
    Result.PrintResult(Y_test, predictions)

if __name__ == "__main__":
    
    X_train, X_test, Y_train, Y_test = ReadSUSYData()
    # lr = LogisticRegression()
    # TrainAndTest( lr, X_train, X_test, Y_train, Y_test)

    # bp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(30,), random_state=1)
    # TrainAndTest( bp, X_train, X_test, Y_train, Y_test)

    clf = DecisionTreeClassifier()
    TrainAndTest( clf, X_train, X_test, Y_train, Y_test)
    featureName = column_names.remove("label")
    # tr = export_text(clf, feature_names=featureName)
    # print(tr)