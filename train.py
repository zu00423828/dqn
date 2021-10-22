from model.net import Net
import torch
from torch import nn
from torch import optim
from data.sim_env import ENV
import numpy as np
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
def save_model():
    torch.save(net.state_dict(),'checkpoint/model.pth')
    torch.save(optimizer.state_dict(),'checkpoint/optim.pth')
def load_model():
    net.load_state_dict(torch.load('checkpoint/model.pth'))
    optimizer.load_state_dict(torch.load('checkpoint/optim.pth'))
def train(use_checkpoint=False,use_cuda=False):
    
    with SummaryWriter() as writer:
        for epoch in range(max_epoch):
            net.train()
            if use_checkpoint:
                load_model()
            optimizer.zero_grad()
            pro_bar=tqdm(dl,desc=f'epoch:{epoch+1}')
            for step,(x,y) in enumerate(pro_bar):
                now_step=epoch*len(dl)+step
                x=x.to(device)
                pre=net(x)
                loss=loss_func(pre,y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if step%100==0:
                    writer.add_scalar('loss',loss.item(),now_step)
            eval() 
            save_model()

def eval():
    net.eval()
    with torch.no_grad():
        origin=np.random.rand(5).astype(np.float32)
        x=origin.copy()
        y=np.zeros((4))
        label=np.argmin(abs(np.delete(origin,2)-origin[2]),axis=0)
        y[label]=1
        x_t=torch.from_numpy(x).unsqueeze(0)
        pre=net(x_t)
        print(x,y.argmax(),pre.argmax(1))
    
    
if __name__ == '__main__':
    n_actions = 4
    n_states = 5
    n_hidden = 50
    batch_size = 100
    lr = 0.0001
    max_epoch = 4000
    use_cuda=False
    dataset=ENV(4,5)

    device=torch.device('cuda' if use_cuda else 'cpu')
    dl=DataLoader(dataset,batch_size=batch_size,drop_last=True,pin_memory=use_cuda)
    net=Net(5,4,60).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.BCELoss().to(device)
    train() 