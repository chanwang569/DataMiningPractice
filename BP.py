# coding:UTF-8
import numpy as np
from math import sqrt
import Utils
import Result
def partial_sig(x):
    m, n = np.shape(x)
    out = np.mat(np.zeros((m, n)))
    for i in range(m):
        for j in range(n):
            out[i, j] = Utils.SigmoidFunction(x[i, j]) * (1 - Utils.SigmoidFunction(x[i, j]))
    return out

def Input2Hidden(iFeatureMatrix, iWeightMatrix0, iBiasMatrix0):
    m = np.shape(iFeatureMatrix)[0]
    hidden_in = iFeatureMatrix * iWeightMatrix0
    for i in range(m):
        hidden_in[i, ] += iBiasMatrix0
    return hidden_in

def HiddenApplySigmoid(iHiddenInputMatrix):
    hidden_output = Utils.SigmoidFunction(iHiddenInputMatrix)
    return hidden_output;

def Hidden2Output(hidden_out, w1, b1):
    m = np.shape(hidden_out)[0]
    predict_in = hidden_out * w1
    for i in range(m):
        predict_in[i, ] += b1
    return predict_in
    
def TrainBPModel(iFeatureMat, iLabelMat, iHiddenLayerNodeCount, iMaxCycle, iLearnRate, iOutputCount, iPrintErrorIter):
    m, n = np.shape(iFeatureMat)
    # Init
    w0 = np.mat(np.random.rand(n, iHiddenLayerNodeCount))
    w0 = w0 * (8.0 * sqrt(6) / sqrt(n + iHiddenLayerNodeCount)) - \
    np.mat(np.ones((n, iHiddenLayerNodeCount))) * (4.0 * sqrt(6) / sqrt(n + iHiddenLayerNodeCount))
    b0 = np.mat(np.random.rand(1, iHiddenLayerNodeCount))
    b0 = b0 * (8.0 * sqrt(6) / sqrt(n + iHiddenLayerNodeCount)) - \
    np.mat(np.ones((1, iHiddenLayerNodeCount))) * (4.0 * sqrt(6) / sqrt(n + iHiddenLayerNodeCount))
    w1 = np.mat(np.random.rand(iHiddenLayerNodeCount, iOutputCount))
    w1 = w1 * (8.0 * sqrt(6) / sqrt(iHiddenLayerNodeCount + iOutputCount)) - \
     np.mat(np.ones((iHiddenLayerNodeCount, iOutputCount))) * (4.0 * sqrt(6) / sqrt(iHiddenLayerNodeCount + iOutputCount))
    b1 = np.mat(np.random.rand(1, iOutputCount))
    b1 = b1 * (8.0 * sqrt(6) / sqrt(iHiddenLayerNodeCount + iOutputCount)) - \
     np.mat(np.ones((1, iOutputCount))) * (4.0 * sqrt(6) / sqrt(iHiddenLayerNodeCount + iOutputCount))
    # Train
    i = 0
    while i <= iMaxCycle:
        i += 1  
        # Forward calculate output
        lHiddenIn =  Input2Hidden(iFeatureMat, w0, b0)
        lHiddenOut = Utils.SigmoidFunction(lHiddenIn)
        loutputIn = Hidden2Output(lHiddenOut, w1, b1)
        lOutputOut = Utils.SigmoidFunction(loutputIn)
        # For update weight. 
        delta_output = -np.multiply((iLabelMat - lOutputOut), partial_sig(loutputIn))
        delta_hidden = np.multiply((delta_output * w1.T), partial_sig(lHiddenIn))
        j = 0
        # Backword update weights       
        w1 = w1 - iLearnRate * (lHiddenOut.T * delta_output)
        b1 = b1 - iLearnRate * np.sum(delta_output, axis=0) * (1.0 / m)
        w0 = w0 - iLearnRate * (iFeatureMat.T * delta_hidden)
        b0 = b0 - iLearnRate * np.sum(delta_hidden, axis=0) * (1.0 / m)
        PrintPara(w0, b0, w1, b1)
        if i % iPrintErrorIter == 0:
            #PrintPara(w0,b0, w1,b1)
            print ("\t-----Trained Count: ", i)
            code = Result.EvaluratePerformance(np.argmax(iLabelMat, axis=1), np.argmax(PredictWithModel(iFeatureMat, w0, w1, b0, b1), axis=1),0.5)
            if code == -1 : 
                j = j + 1
    return w0, w1, b0, b1

def PrintPara(w0, b0, w1, b1):
    print("w0:", w0.max(), "-", w0.min(),"\tb0:",b0.max(), "-", b0.min(), "\tw1:", w1.max(),"-", w1.min(),"\tb1:",b1.max(),"-",b1.min())

def PredictWithModel(feature, w0, w1, b0, b1):
    return Utils.SigmoidFunction(Hidden2Output(Utils.SigmoidFunction(Input2Hidden(feature, w0, b0)), w1, b1))    

def Write2File(file_name, source):   
    f = open(file_name, "w")
    m, n = np.shape(source)
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(str(source[i, j]))
            f.write("\t".join(tmp) + "\n")
    f.close()

def PersistModel(w0, w1, b0, b1):
    Write2File("weight_w0", w0)
    Write2File("weight_w1", w1)
    Write2File("weight_b0", b0)
    Write2File("weight_b1", b1)
    
def LoadBPModel(file_w0, file_w1, file_b0, file_b1):
    def get_model(file_name):
        f = open(file_name)
        model = []
        for line in f.readlines():
            lines = line.strip().split("\t")
            model_tmp = []
            for x in lines:
                model_tmp.append(float(x.strip()))
            model.append(model_tmp)
        f.close()
        return np.mat(model)
    
    w0 = get_model(file_w0)
    w1 = get_model(file_w1)
    b0 = get_model(file_b0)
    b1 = get_model(file_b1)

    return w0, w1, b0, b1