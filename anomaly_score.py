from dataloader import preprocess
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import torch.nn as nn
import torch
import numpy as np

def anomaly_score(z, x):
    """
    Compute the anomaly score of a given data point.
    :param logvar: the log variance of the data point.
    :param sigma: the standard deviation of the data point.
    :param x: the data point.
    :return: the anomaly score of the data point.
    """
    fc = nn.Softmax(dim=0)
    z = torch.Tensor(z).unsqueeze(1)
    x = torch.Tensor(x).unsqueeze(1)
    score = torch.abs(fc(z) - fc(x))
    score = preprocess(score)
    return score

def anomaly_detection(x,recon_x,score,SVR_predicter):
    #train a SVM to detect anomaly
    #return the anomaly score
    x = np.array(x).reshape(-1,1)
    recon_x = np.array(recon_x).reshape(-1,1)
    recon_x = preprocess(recon_x)
    print('=> using an SVR as anomaly score predictor')
    threshold = SVR_predicter.predict(recon_x).reshape(-1,1)
    results = []
    eps = np.linspace(-0.1,0.1,100)
    for epsilon in eps:
        thre = threshold + epsilon
        result = score>thre
        results.append(result)
    return results , threshold

def get_range_proba(predict, label, delay=100):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict