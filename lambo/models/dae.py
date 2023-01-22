import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch
import torch.optim as optim
from torch.autograd import Variable
from lambo.utils import tokens_to_str

class denoising_model(nn.Module):
  def __init__(self):
    super(denoising_model,self).__init__()
    self.encoder=nn.Sequential(
                  nn.Linear(36,18),
                  nn.ReLU(True),
                  nn.Linear(18,9),
                  nn.ReLU(True),
                  nn.Linear(9,5),
                  nn.ReLU(True),
                  nn.Linear(5,2),
                  nn.ReLU(True),
                  )
    
    self.decoder=nn.Sequential(
                  nn.Linear(2,5),
                  nn.ReLU(True),
                  nn.Linear(5,9),
                  nn.ReLU(True),
                  nn.Linear(9,18),
                  nn.ReLU(True),
                  nn.Linear(18,36),
                  )
    
 
  def forward(self,x):
    x=self.encoder(x)
    x=self.decoder(x)
    #x=torch.round(x)
    
    return x

def train_DAE(x, tokenizer):
    x = x.to(torch.float)
    bs = 4
    loader = torch.utils.data.DataLoader(dataset = x,
                                     batch_size = bs,
                                     shuffle = True)
    print("x size",x.shape)

    DAEmodel=denoising_model()
    criterion=nn.MSELoss()
    optimizer=optim.Adam(DAEmodel.parameters(),lr=0.005,weight_decay=1e-9)
    epochs=5000
    outputs = []
    losses = []
    # tok_norm = torch.zeros((bs,36))
    for epoch in range(epochs):
        for tok in loader:
            # Output of Autoencoder
            # mean = torch.mean(tok, 1)
            # std = torch.std(tok, 1)
            # for i in range(bs):
            #     tok_norm[i] = (tok[i] - mean[i]) / std[i]
            reconstructed = DAEmodel(tok)
            # for i in range(bs):
            #     reconstructed[i] = torch.ceil((reconstructed[i] + mean[i])*std[i])
            
            # Calculating the loss function
            loss = criterion(reconstructed, tok)
            
            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Storing the losses in a list for plotting
            losses.append(loss)
        if epoch % 50 == 0:
            print("epoch: ",epoch)
            # print("reconstructed",tokens_to_str(reconstructed, tokenizer))
            print("tok",tokens_to_str(tok, tokenizer))
            print("latest loss", loss)
        outputs.append((epochs, tok, reconstructed))
    return DAEmodel, losses, outputs