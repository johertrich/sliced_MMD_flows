# This code belongs to the paper
#
# J. Hertrich, C. Wald, F. Altekrüger and P. Hagemann (2024)
# Generative Sliced MMD Flows with Riesz Kernels.
# International Conference on Learning Representations.
#
# It reproduces the FashionMNIST example from Section 5.

import torch
import torch.nn.functional as F
import torchvision.datasets as td
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm

from utils.unet import UNet
import utils.utils as ut

device='cuda'
dtype=torch.float

parser = argparse.ArgumentParser()
parser.add_argument('--visualize', type = bool, default = False,
                    help='Visualize the generated samples')
parser.add_argument('--save', type = bool, default = True,
                    help='Save images of particles during training')
args = parser.parse_args()

def vis():
    N=100
    batch_size = 100
    x_new = torch.rand((N,channel,img_size,img_size),dtype=dtype,device=device)
    for n in tqdm(range(len(os.listdir(f'{dir_name}/nets')))):
        new_net = get_UNET()
        new_net.load_state_dict(torch.load(f'{dir_name}/nets/net{n}.pt'))
        x_old = torch.tensor([],device=device,dtype=dtype)
        for i in range(N//batch_size):
            x_tmp = x_new[i*batch_size:(i+1)*batch_size,...]
            out_tmp = x_tmp - new_net(x_tmp).detach()
            x_old = torch.cat([x_old,out_tmp],dim=0)
            
        x_new = x_old.clone()
    ut.save_image(x_new,f'{dir_name}/FashionMNIST_samples.png',10)
    exit()

def get_UNET(input_h=32):
    return UNet(
        input_channels=channel,
        input_height=input_h,
        ch=32,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(256,),
        resamp_with_conv=True,).to(device) 
        
if __name__ == '__main__':
    dir_name='FashionMNIST'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    if not os.path.isdir(dir_name+'/nets'):
        os.mkdir(dir_name+'/nets')

    # Set parameters
    M = 20000
    n_projections = 1000
    channel = 1
    img_size = 28
    momentum = 0.8
    step_size = 1.
    d = channel * img_size**2
    s_factor = ut.sliced_factor(d)
    
    new_net=get_UNET()
    train_steps = 2000
    batch_size = 100
    net_num = 0
    
    step=0
    step_exp = 5
    opt_steps = 2**step_exp

    if args.visualize:
        vis()
    
    #load target samples
    fmnist = td.FashionMNIST('fashionMNIST',transform=transforms.ToTensor(),download=True)
    data = DataLoader(dataset=fmnist,batch_size=M)
    y = next(iter(data))[0].view(M,-1).to(device)

    x = torch.rand((M,d),dtype=dtype,device=device)
    old_grad = torch.zeros((M,d), device = device)

    while True:
        x_old=torch.clone(x)
        for _ in tqdm(range(opt_steps)):
            #draw projections
            xi = torch.randn((n_projections,d),dtype=dtype,device=device)
            xi = xi/torch.sqrt(torch.sum(xi**2,-1,keepdim=True))
            xi = xi.unsqueeze(1)
            
            #slice particles
            x_proj = F.conv1d(x.reshape(1,1,-1),xi,stride=d).reshape(n_projections,-1)
            y_proj = F.conv1d(y.reshape(1,1,-1),xi,stride=d).reshape(n_projections,-1)
            
            #compute 1D gradient of MMD
            grad = ut.MMD_derivative_1d(x_proj,y_proj)
            grad = grad.transpose(0,1)
            
            #compute MMD gradient based on 1D gradient
            xi = xi.reshape([n_projections,d]).transpose(0,1).flatten()
            MMD_grad = s_factor* F.conv1d(xi.reshape([1,1,-1]), grad.unsqueeze(1),
                            stride=n_projections).squeeze()/n_projections   
            MMD_grad = MMD_grad + momentum*old_grad
            
            #update particles
            x -= step_size*M*MMD_grad
            old_grad = MMD_grad
            step=step+1

        #train network
        many_grad = (x_old-x).view(-1,channel,img_size,img_size)
        optim = torch.optim.Adam(new_net.parameters(), lr=0.001)
        for ts in range(train_steps):
            perm = torch.randperm(many_grad.shape[0])[:batch_size]
            y_in = many_grad[perm]
            x_in = x_old[perm].view(batch_size,channel,img_size,img_size)
            loss = torch.sum((new_net(x_in)-y_in)**2)/batch_size
            optim.zero_grad()
            loss.backward()
            optim.step()
        torch.save(new_net.state_dict(),f'{dir_name}/nets/net{net_num}.pt')
        net_num += 1
        x_old = x_old.reshape(M,channel,img_size,img_size)
        
        #update particles
        with torch.no_grad():
            x_new = []
            i = 0
            while i<M:
                x_in = x_old[i:i+batch_size]
                x_new.append(x_in-new_net(x_in).detach())
                i += batch_size
            x_new = torch.cat(x_new,0)
        x = x_new.reshape(M,-1).detach()
        opt_plus = min(2**step_exp,2048)
        opt_steps=min(opt_steps+opt_plus,50000)
        step_exp+=1
        
        if args.save:
            ut.save_image(x_new[:100],f'{dir_name}/flow_net{net_num}.png',nrow=10)
        
