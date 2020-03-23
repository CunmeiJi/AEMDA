import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class regression_model(nn.Module):
    def __init__(self, embed, input_size=128*2):
        super(regression_model, self).__init__()
        self.embed = embed
        self.sim = nn.CosineSimilarity()
        self.dis = nn.PairwiseDistance(p=2)
    
    def forward(self, x):
        x = x.view(-1, 2)
        idx_d1, idx_d2 = x[:,0], x[:,1]
        x_embed_d1,x_embed_d2 = self.embed(idx_d1.long()), self.embed(idx_d2.long())
        x = 0.5 + 0.5*self.sim(x_embed_d1, x_embed_d2)
        return x

class AE(nn.Module):
    def __init__(self,input_size):
        super(AE, self).__init__()
        self.input_size = input_size
        self.fc1  = nn.Linear(input_size, input_size//2)
        self.fc2  = nn.Linear(input_size//2, input_size//8)
        self.fc3  = nn.Linear(input_size//8, 32)
        self.fc31 = nn.Linear(32, input_size//8)
        self.fc21 = nn.Linear(input_size//8, input_size//2)
        self.fc11 = nn.Linear(input_size//2, input_size)
    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        self.z = x
        return x

    def decode(self, x):
        x = F.relu(self.fc31(x))
        x = F.relu(self.fc21(x))
        x = F.tanh(self.fc11(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class dis_model(nn.Module):
    def __init__(self, embed_size=128, dis_size=383):
        super(dis_model,self).__init__()
        self.embed_dis = nn.Embedding(dis_size, embed_size) 
        self.init_weights()
        self.model = regression_model(self.embed_dis, input_size = embed_size*2)

    def init_weights(self):
        initrange = 0.1
        self.embed_dis.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):
        return self.model.forward(x)
        
class mi_model(nn.Module): 
    def __init__(self, embed_size=128, mi_size=495):
        super(mi_model,self).__init__()
        self.embed_mi  = nn.Embedding(mi_size,embed_size)
        self.init_weights()
        self.model = regression_model(self.embed_mi, input_size = embed_size*2)

    def init_weights(self):
        initrange = 0.1
        self.embed_mi.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x):
        return self.model.forward(x)
        
class Net(nn.Module):
    def __init__(self, embed_dis, embed_mi, embed_size=128):
        super(Net,self).__init__()
        self.embed_dis = embed_dis
        self.embed_mi  = embed_mi
        self.ae = AE(embed_size * 2)

    def forward(self, x, device="cpu"):
        x = x.view(-1, 2)
        idx_mi, idx_dis = x[:,0], x[:,1]
        x_embed_dis,x_embed_mi = self.embed_dis(idx_dis.long()), self.embed_mi(idx_mi.long())
        x = torch.cat((x_embed_dis.detach(), x_embed_mi.detach()), 1).to(device)
        x = self.ae(x)
        return x

    def loss_function(self, recon_x, x, device="cpu", lamda = 1e-4):
        x = x.view(-1, 2)
        idx_mi, idx_dis = x[:,0], x[:,1]
        x_embed_dis,x_embed_mi = self.embed_dis(idx_dis.long()), self.embed_mi(idx_mi.long())
        x = torch.cat((x_embed_dis, x_embed_mi), 1).to(device)
        mse_loss = nn.MSELoss(reduction='sum')(recon_x, x.detach()).cpu()

        W1 = self.ae.state_dict()['fc1.weight'].cpu() 
        W2 = self.ae.state_dict()['fc2.weight'].cpu()
        W3 = self.ae.state_dict()['fc3.weight'].cpu()
        con_loss = torch.sum(torch.mm(torch.mm(W3, W2), W1)**2)
        
        return mse_loss + lamda * con_loss

