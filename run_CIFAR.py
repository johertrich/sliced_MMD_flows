# This code belongs to the paper
#
# J. Hertrich, C. Wald, F. Altekrüger and P. Hagemann (2024)
# Generative Sliced MMD Flows with Riesz Kernels.
# International Conference on Learning Representations.
#
# It reproduces the CIFAR10 example from Section 5.

import torch
import torch.nn.functional as F
import torchvision.datasets as td
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os 

from utils.unet import UNet
import utils.utils as ut

parser = argparse.ArgumentParser()
parser.add_argument('--visualize', type = bool, default = False,
                    help='Visualize the generated samples')
parser.add_argument('--save', type = bool, default = True,
                    help='Save images of particles during training')
args = parser.parse_args()

device='cuda'
dtype=torch.float

def vis():
    N=100
    batch_size = 100
    net_changes = torch.load(f'{dir_name}/net_changes.pt')
    x_new = torch.rand((N,channel,img_size//down_factors[0],img_size//down_factors[0]),dtype=dtype,device=device)
    net_num = 0
    tmp = 0 
    factor_num = 0
    for n in tqdm(range(len(os.listdir(f'{dir_name}/nets')))):
        if net_num in net_changes:
            factor_num += 1
        input_h = img_size//down_factors[factor_num]
        net = get_UNET(input_h=input_h)
        net.load_state_dict(torch.load(f'{dir_name}/nets/net{net_num}.pt',map_location=device))
        net.eval()
        x_old = torch.tensor([],device=device,dtype=dtype)
        for i in range(N//batch_size):
            x_tmp = x_new[i*batch_size:(i+1)*batch_size,...]
            out_tmp = x_tmp - net(x_tmp).detach()
            x_old = torch.cat([x_old,out_tmp],dim=0)
        net_num = net_num+1
        x_new = x_old.clone()
        if net_num in net_changes and down_factors[factor_num]>1:
            x_new = x_new.reshape(N,channel,input_h,1,input_h,1).tile(1,1,1,2,1,2).reshape(N,channel,2*input_h,2*input_h)
            x_new += .06*torch.randn_like(x_new)
    ut.save_image(x_new,f'{dir_name}/CIFAR_samples.png',10)
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
    dir_name='CIFAR'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    if not os.path.isdir(dir_name+'/nets'):
        os.mkdir(dir_name+'/nets')

    # Set parameters
    M = 30000
    patch_projections = 500
    proj_batches=1
    channel = 3
    img_size = 32
    momentum = 0.0
    down_factors = [8,4,2,1]
    num_steps = [600000,600000,600000,1000000]
    step_sizes = [.5,1,1.,1.]
    train_steps = 5000
    batch_size = 100
    net_num = 0
    net_changes = []

    step_exp = 5
    opt_steps = 2**step_exp

    if args.visualize:
        vis()
    
    #load target samples
    cifar = td.CIFAR10('cifar10',transform=transforms.ToTensor(),download=True)
    data = DataLoader(dataset=cifar,batch_size=M)
    data = next(iter(data))[0].view(M,-1).to(device)
    
    p_size = 7 #local size for projections
    p_dim = channel * p_size**2
    cut_p = []
    for i in range(4):
        cut_p.append(ut.cut_patches_periodic_padding(img_size,img_size,channel,(i+1)*p_size))
    scales = [0,1,2,3]

    #pyramidal approach
    x = torch.rand((M,channel*(img_size//down_factors[0])**2),dtype=dtype,device=device)
    for factor_num,down_factor in enumerate(down_factors):
        #downsample target
        y = F.avg_pool2d(data.reshape(M,channel,img_size,img_size),down_factor).reshape(M,-1)
        
        cur_size = img_size//down_factor
        d = y.shape[-1]
        step_size = step_sizes[factor_num]
        s_factor = ut.sliced_factor(d)
        n_projections = max(500,d)
        step = 0
        new_net = get_UNET(input_h=cur_size)
        while True:
            old_grad = torch.zeros((M,d), device = device)
            x_old = torch.clone(x)
            for _ in tqdm(range(opt_steps)):
                if cur_size < 32:
                    MMD_grad = 0.
                    for _ in range(proj_batches):
                        #fully-connected projections
                        xi = torch.randn((n_projections,d),dtype=dtype,device=device)
                        xi = xi/torch.sqrt(torch.sum(xi**2,-1,keepdim=True))
                        xi = xi.unsqueeze(1)
                        
                        #slice the particles
                        x_proj = F.conv1d(x.reshape(1,1,-1),xi,stride=d).reshape(n_projections,-1)
                        y_proj = F.conv1d(y.reshape(1,1,-1),xi,stride=d).reshape(n_projections,-1)
                        
                        #compute 1D gradient of MMD
                        grad = ut.MMD_derivative_1d(x_proj,y_proj)
                        grad = grad.transpose(0,1)
                        
                        #compute MMD gradient based on 1D gradient
                        xi = xi.reshape([n_projections,d]).transpose(0,1).flatten()
                        MMD_grad = s_factor* F.conv1d(xi.reshape([1,1,-1]), grad.unsqueeze(1),
                                    stride=n_projections).squeeze()/n_projections + MMD_grad
                else:
                    momentum = 0
                    MMD_grads = torch.zeros(M*d,device=device)
                    for s in scales:
                        MMD_grad = 0.
                        for _ in range(proj_batches):
                            #locally-connected projections
                            xi_l = torch.randn((patch_projections,p_dim),dtype=dtype,device=device)
                            xi_l = xi_l.reshape(-1,channel,p_size,1,p_size,1).tile(1,1,(s+1),1,(s+1)).reshape(-1,channel*((s+1)*p_size)**2)
                            xi_l = xi_l/torch.sqrt(torch.sum(xi_l**2,-1,keepdim=True))
                            xi = xi_l.unsqueeze(1)
                            
                            #extract patches
                            position_inds_height = torch.randint(0,cur_size,(1,),device=device)
                            position_inds_width = torch.randint(0,cur_size,(1,),device=device)
                            patches_x,linear_inds = cut_p[s](x.reshape(-1,channel,cur_size,cur_size),
                                                        position_inds_height,position_inds_width)
                            patches_y,linear_inds = cut_p[s](y.reshape(-1,channel,cur_size,cur_size),
                                                        position_inds_height,position_inds_width)
                            
                            #slice the particles
                            x_proj = F.conv1d(patches_x.reshape(1,1,-1),xi,
                                        stride=channel*((s+1)*p_size)**2).reshape(patch_projections,-1)
                            y_proj = F.conv1d(patches_y.reshape(1,1,-1),xi,
                                        stride=channel*((s+1)*p_size)**2).reshape(patch_projections,-1)
                                        
                            #compute 1D gradient of MMD
                            grad = ut.MMD_derivative_1d(x_proj,y_proj)
                            grad = grad.transpose(0,1)
                            
                            #compute MMD gradient based on 1D gradient
                            xi = xi.reshape([patch_projections,channel*((s+1)*p_size)**2]).transpose(0,1).flatten()
                            MMD_grad = s_factor* F.conv1d(xi.reshape([1,1,-1]), grad.unsqueeze(1),
                                            stride=patch_projections).squeeze()/patch_projections + MMD_grad
                        MMD_grads[linear_inds] += MMD_grad.reshape(-1)
                    MMD_grad = MMD_grads.reshape(M,-1) 
                MMD_grad = MMD_grad/proj_batches + momentum*old_grad
                
                #update the flow
                x -= step_size*M*MMD_grad
                old_grad = MMD_grad
                step=step+1
            
            #train network
            many_grad = (x_old-x).view(-1,channel,cur_size,cur_size)
            optim = torch.optim.Adam(new_net.parameters(), lr=0.0005)
            for ts in range(train_steps):
                perm = torch.randperm(many_grad.shape[0])[:batch_size]
                y_in = many_grad[perm]
                x_in = x_old[perm].view(batch_size,channel,cur_size,cur_size)
                loss = torch.sum(torch.abs(new_net(x_in)-y_in))/batch_size
                optim.zero_grad()
                loss.backward()
                optim.step()
            torch.save(new_net.state_dict(),f'{dir_name}/nets/net{net_num}.pt')
            net_num += 1

            #update particles
            x_old = x_old.reshape(M,channel,cur_size,cur_size)
            with torch.no_grad():
                x_new = []
                i = 0
                while i<M:
                    x_in = x_old[i:i+batch_size]
                    x_new.append(x_in-new_net(x_in).detach())
                    i += batch_size
                x_new = torch.cat(x_new,0)
            x = x_new.reshape(M,-1).detach()
            opt_plus=min(2**step_exp,1024)
            opt_steps=min(opt_steps+opt_plus,30000)
            momentum = min(0.8,momentum + 0.01)
            step_exp+=1
                
            if args.save:    
                ut.save_image(x_new[:100],f'{dir_name}/flow_net{net_num}.png',nrow=10)
            
            if step>=num_steps[factor_num]:
                break
        
        #upsample to higher resolution        
        net_changes.append(net_num)
        torch.save(torch.tensor(net_changes),f'{dir_name}/net_changes.pt')
        if down_factor>1:
            x = x.reshape(M,channel,cur_size,1,cur_size,1).tile(1,1,1,2,1,2).reshape(M,channel,2*cur_size,2*cur_size)
            x += .08*torch.randn_like(x)
            x = x.reshape(M,-1)
            
