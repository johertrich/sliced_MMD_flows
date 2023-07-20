# This code belongs to the paper
#
# J. Hertrich, C. Wald, F. Altekrüger and P. Hagemann (2023)
# Generative Sliced MMD Flows with Riesz Kernels.
# Arxiv Preprint 2305.11463
# 

import numpy as np
import time
import matplotlib.pyplot as plt
from utils.MMD_1D_der import *
import torchvision.datasets as td
from torchvision.utils import make_grid
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import pickle
import copy
from utils.unet import UNet

device='cuda'
dtype=torch.float
dir_name='MNIST_results'

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
if not os.path.isdir(dir_name+'/nets'):
    os.mkdir(dir_name+'/nets')

def get_UNET():
    # use unet
    return UNet(
        input_channels=1,
        input_height=28,
        ch=32,
        ch_mult=(1, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(256,),
        resamp_with_conv=True,).to(device) 


def save_image(trajectory,name,nrow=25):
    grid = make_grid(trajectory,nrow=nrow,padding=1,pad_value=.5)
    plt.imsave(f'{dir_name}/{name}.png',torch.clip(grid.permute(1,2,0),0,1).cpu().numpy())
    return
    
# Load target samples 
mnist = td.MNIST('mnist',transform=transforms.ToTensor(),download=True)
M=20000
data = DataLoader(dataset=mnist,batch_size=M)
y = next(iter(data))[0].view(M,28**2).to(device)
step_size=1.
momentum=0.
pickle.dump(y.detach().cpu().numpy(),open('imgs5/target.pickle',"wb"))

# Set parameters
N=M
momentum = 0.85
d=y.shape[-1]
s_factor=sliced_factor(d)
n_projections=1200
opt_steps=500
train_steps = 3001
batch_size=100

# initialize variables
x=torch.rand((N,d),dtype=dtype,device=device)
step=0
trajectory = torch.empty(0,device=device)
tic=time.time()
x=torch.rand((N,1,28,28),dtype=dtype,device=device).reshape(N,-1)
x_test=torch.rand((100,1,28,28),dtype=dtype,device=device)
new_net=get_UNET()
old_grad = torch.zeros((N,d), device = device)
net_num=0
first = True
while True:
    print('Start',step,time.time()-tic)    
    x_old=torch.clone(x)
    print('Samples computed',step,time.time()-tic)
    # Compute the steps of the flow
    for _ in range(opt_steps):
        xi=torch.randn((n_projections,d),dtype=dtype,device=device)
        xi=xi/torch.sqrt(torch.sum(xi**2,-1,keepdim=True))
        xi=xi.unsqueeze(1)
        x_proj=torch.nn.functional.conv1d(x.reshape(1,1,-1),xi,stride=d).reshape(n_projections,-1)
        y_proj=torch.nn.functional.conv1d(y.reshape(1,1,-1),xi,stride=d).reshape(n_projections,-1)
        grad=MMD_derivative_1d(x_proj,y_proj)
     
        grad = grad.transpose(0,1)
        xi = xi.reshape([n_projections,d]).transpose(0,1).flatten()
        MMD_grad = s_factor* torch.nn.functional.conv1d(xi.reshape([1,1,-1]), grad.unsqueeze(1),stride=n_projections).squeeze()/n_projections   

        MMD_grad = MMD_grad + momentum*old_grad
        x=x-step_size*N*MMD_grad
        old_grad = MMD_grad
        if (step+1)%2000==0:
            print('Compute flow',step,time.time()-tic)
        step=step+1

    print('Train network',step,time.time()-tic)
    # difference to approximate
    many_grad=(x_old-x).view(-1,1,28,28)
    # train network Phi_l
    optim = torch.optim.Adam(new_net.parameters(), lr=0.001)
    loss_sum=0.
    for ts in range(train_steps):
        perm=torch.randperm(many_grad.shape[0])[:batch_size]
        y_in=many_grad[perm]
        x_in=x_old[perm].view(batch_size,1,28,28)
        loss=torch.sum((new_net(x_in)-y_in)**2)/batch_size
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_sum+=loss.item()
        if (ts+1)%100==0:
            print(ts+1,train_steps,loss.item(),loss_sum/(ts+1))
    torch.save(new_net.state_dict(),'imgs5/nets/net'+str(net_num)+'.pt')
    net_num+=1
    x_old=x_old.reshape(N,1,28,28)

    # Compute samples for the next network
    with torch.no_grad():
        x_new=[]
        i=0
        while i<N:
            x_in=x_old[i:i+batch_size]
            x_new.append(x_in-new_net(x_in).detach())
            i+=batch_size
        x_new=torch.cat(x_new,0)
        x_test=x_test-new_net(x_test)
        x_test=x_test.detach()
    save_image(torch.cat((x_new[:100],x_old[:100].reshape(100,1,28,28),x[:100].reshape(100,1,28,28)),0),'mnist'+str(step),nrow=10)
    save_image(x_test,'mnist_test'+str(step),nrow=10)
    x=x_new.reshape(N,-1).detach()
    print('Network trained',step,time.time()-tic)
    pickle.dump(x.detach().cpu().numpy(),open('imgs5/samples.pickle',"wb"))
    # update parameters
    opt_steps= min(1000+opt_steps,30000)
    if step>=4000001:
        break

