# coding:UTF-8
import numpy as np
import os
import Result
import Utils

def TrainModel(iFeatureMatrix, iLabelMatrix, iMaxCycleInt, iStudyRateFloat , iPrintErrorIter):
    lColoumnCount = np.shape(iFeatureMatrix)[1]  
    lWeightMatrix = np.mat(np.ones((lColoumnCount, 1))) 
    #lWeightMatrix = np.mat(np.random.rand(lColoumnCount)).T
    lTrainedCount = 0
    while lTrainedCount <= iMaxCycleInt:  
        lTrainedCount += 1 
        lCalculatedLabelMatrix = Utils.SigmoidFunction(iFeatureMatrix * lWeightMatrix)  
        if (lTrainedCount % iPrintErrorIter == 0 ):
            print("\t Trained Count: " + str(lTrainedCount))
            Result.EvaluratePerformance(iLabelMatrix, lCalculatedLabelMatrix,0.5)
            if max(lCalculatedLabelMatrix) == 0:
                iMaxCycleInt += 10
        lWeightMatrix = lWeightMatrix + iStudyRateFloat * iFeatureMatrix.T * (iLabelMatrix - lCalculatedLabelMatrix)  
    return lWeightMatrix


def PredictWithModel(iDataMatrix, iTrainedWeights):
    lPriditedMatix = Utils.SigmoidFunction(iDataMatrix * iTrainedWeights)
    lResultCount = np.shape(lPriditedMatix)[0]
    for i in range(lResultCount):
        if lPriditedMatix[i, 0] < 0.5:
            lPriditedMatix[i, 0] = 0
        else:
            lPriditedMatix[i, 0] = 1
    return lPriditedMatix

def PersistModel(iModelFileName, iWeightMatrix):
    m = np.shape(iWeightMatrix)[0]
    f_w = open(iModelFileName, "w")
    w_array = []
    for i in range(m):
        w_array.append(str(iWeightMatrix[i, 0]))
    f_w.write("\t".join(w_array))
    f_w.close()      