import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from imblearn.under_sampling import RandomUnderSampler
    
import os
import BP
import LR
import CART
import Result
def categorize(df):
    new_df = df.copy()
    le = preprocessing.LabelEncoder()
    
    new_df['job'] = le.fit_transform(new_df['job'])
    new_df['marital'] = le.fit_transform(new_df['marital'])
    new_df['education'] = le.fit_transform(new_df['education'])
    new_df['default'] = le.fit_transform(new_df['default'])
    new_df['housing'] = le.fit_transform(new_df['housing'])
    new_df['month'] = le.fit_transform(new_df['month'])
    new_df['loan'] = le.fit_transform(new_df['loan'])
    new_df['contact'] = le.fit_transform(new_df['contact'])
    new_df['day'] = le.fit_transform(new_df['day'])
    new_df['poutcome'] = le.fit_transform(new_df['poutcome'])
    new_df['y'] = le.fit_transform(new_df['y'])
    return new_df

def remove_outliers(df, column , minimum, maximum):
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values<minimum, col_values>maximum), col_values.mean(), col_values)
    return df

def ReadBankDataSet():
    fileName="bank/bank-full.csv"
    path = os.path.realpath(os.curdir)
    fileNameFullPath = os.path.join(path, fileName)
    data_train = pd.read_csv(fileNameFullPath, na_values =['NA'])
    columns = data_train.columns.values[0].split(';')
    columns = [column.replace('"', '') for column in columns]
    data_train = data_train.values
    data_train = [items[0].split(';') for items in data_train]
    data_train = pd.DataFrame(data_train,columns = columns)
    data_train['job'] = data_train['job'].str.replace('"', '')
    data_train['marital'] = data_train['marital'].str.replace('"', '')
    data_train['education'] = data_train['education'].str.replace('"', '')
    data_train['default'] = data_train['default'].str.replace('"', '')
    data_train['housing'] = data_train['housing'].str.replace('"', '')
    data_train['loan'] = data_train['loan'].str.replace('"', '')
    data_train['contact'] = data_train['contact'].str.replace('"', '')
    data_train['month'] = data_train['month'].str.replace('"', '')
    data_train['day'] = data_train['day'].str.replace('"', '')
    data_train['poutcome'] = data_train['poutcome'].str.replace('"', '')
    data_train['y'] = data_train['y'].str.replace('"', '')

    data_train = data_train[data_train.education != 'unknown']
    data_train = categorize(data_train)

    data= data_train.apply(pd.to_numeric, errors="ignore")
    min_val = data["duration"].min()
    max_val = 1500
    data = remove_outliers(df=data, column='duration' , minimum=min_val, maximum=max_val)

    min_val = data["age"].min()
    max_val = 80
    data = remove_outliers(df=data, column='age' , minimum=min_val, maximum=max_val)

    min_val = data["campaign"].min()
    max_val = 6
    data = remove_outliers(df=data, column='campaign' , minimum=min_val, maximum=max_val)

    data = data.drop('poutcome',axis=1)
    data = data.drop('contact',axis=1)
    return data

def SplitDataset(data , iAppendBias):
    X1 = data.drop('y',axis = 1).values
   
    y = data['y'].values
    under_model = RandomUnderSampler()
    x_under, y_under = under_model.fit_sample(X1,y)
    if iAppendBias:
        beta = np.mat(np.ones((x_under.shape[0], 1)))
        # 加上B 列
        X = np.append(x_under,beta, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(x_under, y_under, test_size=0.25, random_state=42)

    Y_train = np.mat(Y_train).T
    Y_test =np.mat(Y_test).T
    return  X_train, X_test, Y_train,Y_test

def TestBankLRCase():
    data = ReadBankDataSet()
    X_train, X_test, Y_train,Y_test =SplitDataset(data , True)
    w = LR.TrainModel(X_train, Y_train, 1000, 0.1, 100001)
    PredictResult = LR.PredictWithModel(X_train, w)

    Result.PrintResult(Y_train, PredictResult)
    PredictResult2 = LR.PredictWithModel(X_test, w)
    #Result.DrawConfusionMatrix(np.mat(Y_test).T,PredictResult)
    print("Accuracy :\t ", accuracy_score(Y_test, PredictResult2))
    Result.PrintResult(Y_test, PredictResult2)

def TestBankBPCase():
    data = ReadBankDataSet()
    X_train, X_test, Y_train,Y_test =SplitDataset(data, False)
    m = np.shape(X_train)[0]
    lTrainLabel = np.mat(np.zeros((m, 2)))
    for i in range(m):
        lTrainLabel[i, Y_train[i]] = 1
        
    w0, w1, b0, b1 = BP.TrainBPModel(X_train, lTrainLabel, 50 , 5000, 0.1, 2, 20)
    result = BP.PredictWithModel(X_train, w0, w1, b0, b1)
    BP.PersistModel(w0, w1, b0, b1)
    Result.PrintResult(Y_train, np.mat(np.argmax(result,axis = 1)))
    n = np.shape(Y_train)[0]
    lTestLabel = np.mat(np.zeros((n, 2)))

    result2 = BP.PredictWithModel(X_test, w0, w1, b0, b1)
    Result.PrintResult(Y_test,np.mat(np.argmax(result2,axis = 1)))

def TestBankCARTCase():
    data = ReadBankDataSet()
    head = data.head()
    Cloumns =list(data.columns.values)
    X = data.drop('y',axis = 1).values
    Y = data['y'].values
    under_model = RandomUnderSampler()
    x_under, y_under = under_model.fit_sample(X,Y)
    
    X_train, X_test, Y_train,Y_test =train_test_split(x_under, y_under, test_size=0.25, random_state=42)
    
    X_Y_train = X_train.tolist()
    for i in range(len(X_Y_train)):
        X_Y_train[i].append(Y_train[i])
   
    labels  = Cloumns[:]
    desicionTree = CART.TrainCART(X_Y_train, labels)
    #CART.treePlotter.createPlot(desicionTree)
    testData = X_test.tolist()
    result = CART.ClassifyAll(desicionTree, Cloumns, testData)
    Result.PrintResult(Y_test, result)
if __name__ == "__main__":
    TestBankLRCase()
    #TestBankBPCase()
    #estBankCARTCase()
    