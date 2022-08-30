from cmath import inf, log
import encodings
from anomaly_score import anomaly_score, anomaly_detection, get_range_proba
from numpy import argmax, save
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
from dataloader import get_dataloader,preprocess
from model import LSTM_VAE, gauss_sampling
import pathlib
from matplotlib import pyplot as plt
from sklearn.svm import SVR 
from sklearn.model_selection import GridSearchCV
import joblib
import pickle

from zmq import device


def criterion_func(output, target):
    mean1 = output[0]
    logvar1 = output[1]
    mean2 = output[2]
    logvar2 = output[3]
    recon_x = output[4]
    target_ = target
    target = target.view(target.shape[0]*target.shape[1],target.shape[2])
    mean1,logvar1,mean2,logvar2 = mean1.view(mean1.shape[0]*mean1.shape[1],-1),\
    logvar1.view(logvar1.shape[0]*logvar1.shape[1],-1),\
    mean2.view(mean2.shape[0]*mean2.shape[1],-1),\
    logvar2.view(logvar2.shape[0]*logvar2.shape[1],-1)
    
    loss1 = 0
    loss2 = 0
    for i in range(logvar1.shape[0]):
        sigma1 = torch.diag(logvar1[i])
        sigma2 = torch.diag(torch.exp(logvar2[i]))
        mu1 = mean1[i]
        mu2 = mean2[i]
        #print(torch.det(sigma2))
        #print(log(torch.det(sigma2)))
        #print(1/sigma2)
        reconstrc_loss = -0.5*log(torch.det(sigma2))+(target[i]-mu2).t()*(1/sigma2)*(target[i]-mu2)+log(6.28)
        loss1 += reconstrc_loss.squeeze().real

        KLD = -0.5 * torch.sum(1 + sigma1 - mu1.pow(2) - torch.exp(sigma1))
        loss2 += KLD
    loss1 = loss1/logvar1.shape[0]
    loss2 = loss2/logvar1.shape[0]
    MSE =  F.mse_loss(recon_x.reshape(recon_x.shape[0]*recon_x.shape[1],-1) , target,reduction='sum')
    return MSE
    


class Detector(object):
    DEFAULTS = {}           

    def __init__(self, config):

        self.__dict__.update(Detector.DEFAULTS, **config)     #更新默认参数

        self.trainloader = get_dataloader(self.dataname, 'train',self.batch_size,self.seq_lenth)
        self.testloader = get_dataloader(self.dataname, 'test',self.batch_size,self.seq_lenth)
        self.valiloader = get_dataloader(self.dataname, 'vali',self.batch_size,self.seq_lenth)
        self.model = LSTM_VAE(self.input_size, self.hidden_size, self.latent_size, self.dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = criterion_func

        if self.device == 'cuda':
            self.model.cuda()

        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()

    def train(self):
        print('Train:')
        if self.pretrained_epoch is not 0:
            print('Load pretrained model')
            self.model.load_state_dict(torch.load('./model/'+self.dataname+'/model_{}.pth'.format(self.pretrained_epoch -1)))
        for epoch in range(self.pretrained_epoch,self.num_epochs):
            train_loss = 0
            self.model.train()
            start = time.time()
            for i, data in enumerate(self.trainloader):
                inputseq = data[0]
                targetseq = data[1]
                inputseq = inputseq.to(self.device)
                targetseq = targetseq.to(self.device)
                self.model.zero_grad()
                output = self.model(inputseq.to(torch.float32))
                loss = self.criterion(output,targetseq.to(torch.float32))
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                if i % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch+1, self.num_epochs, i+1, len(self.trainloader), loss.item()))
            print('Epoch [{}/{}], Loss: {:.4f}, Time: {:.4f}'
                    .format(epoch+1, self.num_epochs, train_loss, time.time()-start))
            
            vali_loss = self.vali()
            best_vali_loss = inf
            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
                save_dir = pathlib.Path('model',self.dataname).with_suffix('')
                save_dir.mkdir(parents=True,exist_ok=True)
                torch.save(self.model.state_dict(), str(save_dir.joinpath('model_{}.pth'.format(epoch))))
            print('Epoch [{}/{}], Vali Loss: {:.4f}, Best Vali Loss: {:.4f}'
                    .format(epoch+1, self.num_epochs, vali_loss, best_vali_loss))
            print('-'*100)
        print('Train Finished!')
        print('Train a SVR model to predict anomaly score...')
        with torch.no_grad():
            predictions = []
            targets = []
            for i in range(len(self.trainloader.dataset.value)-1):
                inputseq = torch.Tensor(self.trainloader.dataset.value[i])
                targetseq = torch.Tensor(self.trainloader.dataset.value[i+1])
                inputseq = inputseq.to(self.device).reshape(1,1,-1)
                targetseq = targetseq.to(self.device).reshape(1,1,-1)
                output = self.model(inputseq)
                recon_x = output[4]
                predictions.append(recon_x.view(1))
                targets.append(targetseq.view(1))

            scores = anomaly_score(predictions,targets)
            scores = scores.reshape(-1,1)
            score_predictor = GridSearchCV(SVR(kernel='rbf'), cv=5,param_grid={"C": [1e0, 1e1, 1e2],"gamma": np.logspace(-1, 1, 3)})
            predictions = np.array(predictions).reshape(-1,1)
            predictions = preprocess(predictions)
            scores = preprocess(scores).reshape(scores.shape[0])
            score_predictor.fit(predictions,scores)
            
            save_dir = pathlib.Path('model',self.dataname).with_suffix('')
            save_dir.mkdir(parents=True,exist_ok=True)
            pickle.dump(score_predictor, open(save_dir.joinpath('svr_model.pkl'),'wb'))
            print('SVR model saved!')

    def test(self):
        print('Test:')
        print('Loading model...')
        self.model.load_state_dict(torch.load('./model/'+self.dataname+'/model_{}.pth'.format(self.num_epochs-1)))
        self.model.eval()
        print('calculating anomaly score...')
        with torch.no_grad():
            predictions = []
            targets = []
            for i, data in enumerate(self.testloader):
                inputseq = data[0]
                targetseq = data[1]
                inputseq = inputseq.to(self.device)
                targetseq = targetseq.to(self.device)
                output = self.model(inputseq)
                recon_x = output[4]
                predictions.append(recon_x.view(1))
                targets.append(targetseq.view(1))
            scores = anomaly_score(predictions,targets)
            scores = scores.reshape(-1,1)

            self.SVR = pickle.load(open('./model/'+self.dataname+'/svr_model.pkl','rb'))


            results, threshold = anomaly_detection(targets,predictions,scores,self.SVR)
            label = self.testloader.dataset.label[1:]
            eps = [1e-0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
            p = []
            a = []
            r = []
            f = []
            for result in results:
                idx = result * 2 + label
                tn = (idx == 0.0).sum().item()  # tn
                fn = (idx == 1.0).sum().item()  # fn
                fp = (idx == 2.0).sum().item()  # fp
                tp = (idx == 3.0).sum().item()  # tp
                precision = tp / (tp + fp + 1e-7)
                recall = tp / (tp + fn + 1e-7)
                accuracy = (tp + tn) / (tp + tn + fn + fp)
                f1 = 2 * precision * recall / (precision + recall + 1e-7)
                p.append(precision)
                r.append(recall)
                f.append(f1)
                a.append(accuracy)
                new_result = get_range_proba(result,label,7)
                idx = new_result * 2 + label
                tn = (idx == 0.0).sum().item()  # tn
                fn = (idx == 1.0).sum().item()  # fn
                fp = (idx == 2.0).sum().item()  # fp
                tp = (idx == 3.0).sum().item()  # tp
                precision = tp / (tp + fp + 1e-7)
                recall = tp / (tp + fn + 1e-7)
                accuracy = (tp + tn) / (tp + tn + fn + fp)
                f1 = 2 * precision * recall / (precision + recall + 1e-7)
                p.append(precision)
                r.append(recall)
                f.append(f1)
                a.append(accuracy)
            
            print('Precision: {:.4f}, Recall: {:.4f}, Accuracy: {:.4f}, F1: {:.4f}'
                    .format(p[argmax(f)], r[argmax(f)], a[argmax(f)], f[argmax(f)]))
            print('-'*100)
            log = np.concatenate([scores,result,label],axis=1)

            save_dir = pathlib.Path('result',self.dataname).with_suffix('')
            save_dir.mkdir(parents=True,exist_ok=True)

            np.savetxt('./result/'+self.dataname+'/result_{}.csv'.format(self.num_epochs-1),log,delimiter=',')

        if self.save_fig:
            print('Saving figures...')
            save_dir = pathlib.Path('result',self.dataname).with_suffix('').joinpath('fig_detection')
            save_dir.mkdir(parents=True,exist_ok=True)

            fig, ax1 = plt.subplots(figsize=(15,5))
            #ax1.plot(self.testloader.dataset.target[1:],label='Target',
             #        color='black',  marker='.', linestyle='--', markersize=1, linewidth=0.5)
            ax1.plot(predictions, label='predictions',
                     color='blue', marker='.', linestyle='dashed', markersize=1, linewidth=0.5)
            ax1.plot(self.testloader.dataset.value[1:].reshape(-1,1), label='ground truth',
                     color='green', marker='.', linestyle='solid', markersize=1, linewidth=0.5)
            ax1.legend(loc='upper left')
            ax1.set_ylabel('Value',fontsize=15)
            ax1.set_xlabel('Index',fontsize=15)
            ax2 = ax1.twinx()
            ax2.plot(scores, label='Anomaly scores',
                     color='red', marker='.', linestyle='dashed', markersize=1, linewidth= 0.5)
            ax2.plot(threshold,label = 'Threshold',
                    color = 'brown',marker = '.',linestyle = 'dotted',markersize = 1,linewidth = 1)
            #if args.compensate:
                #ax2.plot(predicted_score, label='Predicted anomaly scores from SVR',
                         #color='cyan', marker='.', linestyle='--', markersize=1, linewidth=1)
           #ax2.plot(score.reshape(-1,1)/(predicted_score+1),label='Anomaly scores from \nmultivariate normal distribution',
            #        color='hotpink', marker='.', linestyle='dashed', markersize=1, linewidth=1)
            ax2.legend(loc='upper right')
            ax2.set_ylabel('anomaly score',fontsize=15)
            #plt.axvspan(2830,2900 , color='yellow', alpha=0.3)
            plt.title('Anomaly Detection on ' + self.dataname + ' Dataset', fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.xlim([0,len(self.testloader.dataset.value)-1])
            plt.savefig(str(save_dir.joinpath('fig_scores').with_suffix('.png')))
            #plt.show()
            plt.close()
            print('Done!')


    def vali(self):
        self.model.eval()
        with torch.no_grad():
            vali_loss = 0
            for i, data in enumerate(self.valiloader):
                inputseq = data[0]
                targetseq = data[1]
                inputseq = inputseq.to(self.device)
                targetseq = targetseq.to(self.device)
                output = self.model(inputseq)
                loss = self.criterion(output,targetseq)
                vali_loss += loss.item()
            vali_loss = vali_loss/len(self.valiloader)
        return vali_loss
        