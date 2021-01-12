import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import pylab as pl
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def CalculateConfusionMatrix(actual, predicted, threshold):
    if len(predicted) != len(actual): return -1
    recordCount = len(predicted)
    lActual = actual.astype('int')
    lPredict = predicted.astype('int')
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    for i in range(len(predicted)):
        if actual[i] > 0.5: #正样本
            if predicted[i] > threshold:
                tp += 1.0 #被正确分类的正例
            else:
                fn += 1.0 #本来是正例，错分为负例
        else:              #负样本
            if predicted[i] <= threshold:
                tn += 1.0 #被正确分类的负例
            else:
                fp += 1.0 #本来是负例，被错分为正例

    rtn = [tp/recordCount, fn/recordCount, fp/recordCount, tn/recordCount]
    return rtn
def printConfusionMatic(AcutualLabel, PredictLabel, threshold):
    conMatTest = CalculateConfusionMatrix(AcutualLabel, PredictLabel, threshold)
    tp = conMatTest[0]; fn = conMatTest[1]; fp = conMatTest[2]; tn = conMatTest[3]
    print("\t tp = " + str(tp) + "\tfn = " + str(fn) + "\t" + "fp = " + str(fp) + "\t tn = " + str(tn) + '\n')

def DrawROC(AcutualLabel, PredictLabel):
    printConfusionMatic(AcutualLabel, PredictLabel, 0.5)
    fpr, tpr, thresholds = roc_curve(AcutualLabel.astype('int'),PredictLabel.astype('int'),pos_label=1)
    print("\t tpr=" + str(tpr) + "\t fpr=" + str(fpr) )
    roc_auc = auc(fpr, tpr)
  
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Out-of-sample ROC rocks versus mines')
    pl.legend(loc="lower right")
    pl.show()

def EvaluratePerformance(iAcutalLebelArray, iPredictLabelArray,iThreshold):
    lProcessedPridictLabelArray =[]
    for i in range(len(iPredictLabelArray)):
        if iPredictLabelArray[i] > iThreshold:
            lProcessedPridictLabelArray.append(1)
        else:
            lProcessedPridictLabelArray.append(0)
    
    print("Accuracy:",metrics.accuracy_score(iAcutalLebelArray, lProcessedPridictLabelArray))
    if max(lProcessedPridictLabelArray) > 0:
        print("Precision:",metrics.precision_score(iAcutalLebelArray, lProcessedPridictLabelArray))
        print("Recall:",metrics.recall_score(iAcutalLebelArray, lProcessedPridictLabelArray))
    else:
        return -1

def DrawConfusionMatrix(iAcutalLebelArray, iPredictLabelArray): 
    EvaluratePerformance(iAcutalLebelArray,iPredictLabelArray, 0.5)
    lConfusionMatrix = metrics.confusion_matrix(iAcutalLebelArray, iPredictLabelArray)
    lConfusionMatrix
    class_names=[0,1] # name  of classes
    lFigure, lAxies = plt.subplots()
    lTickMmarks = np.arange(len(class_names))
    plt.xticks(lTickMmarks, class_names)
    plt.yticks(lTickMmarks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(lConfusionMatrix), annot=True, cmap="YlGnBu" ,fmt='g')
    lAxies.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    fpr, tpr, _ = metrics.roc_curve(iAcutalLebelArray, iPredictLabelArray)
    auc = metrics.roc_auc_score(iAcutalLebelArray, iPredictLabelArray)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.show()
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
def PrintResult(Y_test, PredictResult2):
    print("--------------Begin-------------------------")
    print("Accuracy :\t ", accuracy_score(Y_test, PredictResult2))
    print("roc_auc_score :\t",roc_auc_score(Y_test, PredictResult2))
    print("Confusion Matrix : \n",confusion_matrix(Y_test, PredictResult2))
    print("Classification Report: \n",classification_report(Y_test, PredictResult2))
    print("--------------End-------------------------")