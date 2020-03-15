import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from data import disease_name, mirna_name, dis_sim, mi_sim, dis_mi
from model import dis_model, mi_model, Net
from datahelp import DisDataset, MiDataset, KnownDisMiDataset
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import random

class Config(object):
    def __init__(self):
        self.embed_size = 2048
        self.batch_size = 128
        self.epochs = 100
        self.log_interval = 1000
        self.alpha = 0.4
        self.beta = 0.3

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if  torch.cuda.is_available() else {}

opt = Config()

def train_reprenstation(epoch, model, optimizer, train_loader):
    model.train()
    
    train_loss = 0
    for batch_idx, (data,y) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        predict = model(data)
        loss = nn.MSELoss(reduction="sum")(predict, y.to(device)).cpu()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx %  opt.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss

def train(epoch, model, optimizer, train_loader):
    model.train()
    
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data,device)
        loss = model.loss_function(recon_batch, data,device)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if False:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss


if __name__ == '__main__':


	#training representation of diseases
    model_dis = dis_model(opt.embed_size).to(device)
    optimizer_d = optim.Adam(model_dis.parameters(), lr=1e-4)
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d,'min',factor=0.1, patience=4, verbose=True)
    train_loader_dis = DataLoader(dataset=DisDataset(opt.alpha),batch_size= opt.batch_size, shuffle=True, **kwargs)
    
    for epoch in range(1,  opt.epochs+1):
        train_loss = train_reprenstation(epoch,model_dis,optimizer_d,train_loader_dis)
        scheduler_d.step(train_loss) 

    #training representation of miRNAs
    model_mi  = mi_model(opt.embed_size).to(device)
    optimizer_m = optim.Adam(model_mi.parameters(), lr=1e-4)
    scheduler_m = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_m, 'min',factor=0.1, patience=4, verbose=True)
    train_loader_mi = DataLoader(dataset=MiDataset(opt.beta), batch_size= opt.batch_size, shuffle=True, **kwargs)
	
    for epoch in range(1,  opt.epochs+1):
        train_loss = train_reprenstation(epoch, model_mi, optimizer_m, train_loader_mi)
        scheduler_m.step(train_loss) 

	#training predictor
    model     = Net(model_dis.embed_dis, model_mi.embed_mi, embed_size= opt.embed_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=4, verbose=True)
    train_loader = DataLoader(dataset=KnownDisMiDataset(),batch_size= opt.batch_size, shuffle=True, **kwargs)

    for epoch in range(1,  opt.epochs+1):
        train_loss = train(epoch,model,optimizer,train_loader)
        scheduler.step(train_loss) 
