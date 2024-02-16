# This code belongs to the paper
#
# J. Hertrich, C. Wald, F. Altekrüger and P. Hagemann (2024)
# Generative Sliced MMD Flows with Riesz Kernels.
# International Conference on Learning Representations.

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

def MMD_derivative_1d(x,y,only_potential=False):
    '''
    compute the derivate of MMD in 1D
    '''
    N=x.shape[1]
    P=1
    if len(x.shape)>1:
        P=x.shape[0]
    # potential energy
    if y is None:
        grad=torch.zeros(P,N,dtype=dtype,device=device)
    else:
        M=y.shape[1]
        _,inds=torch.sort(torch.cat((x,y),1))
        grad=torch.where(inds>=N,1.,0.).type(dtype)
        grad=(2*torch.cumsum(grad,-1)-M) / (N*M)
        _,inverted_inds=torch.sort(inds)
        inverted_inds=inverted_inds[:,:N]+torch.arange(P,device=device).unsqueeze(1)*(N+M)
        inverted_inds=torch.flatten(inverted_inds)
        grad=grad.flatten()
        grad=grad[inverted_inds].reshape(P,-1)


    if not only_potential:
        _,inds_x=torch.sort(x)
        inds_x=inds_x+torch.arange(P,device=device).unsqueeze(1)*N
        inds_x=torch.flatten(inds_x)
        # interaction energy
        interaction=2*torch.arange(N,dtype=dtype,device=device)-N+1
        interaction=(1/(N**2)) * interaction
        interaction=interaction.tile(P,1)
        grad=grad.flatten()
        grad[inds_x]=grad[inds_x]-interaction.flatten()
        grad=grad.reshape(P,-1)

    return grad

def sliced_factor(d):
    '''
    compute the scaling factor of sliced MMD
    '''
    k=(d-1)//2
    fac=1.
    if (d-1)%2==0:
        for j in range(1,k+1):
            fac=2*fac*j/(2*j-1)
    else:
        for j in range(1,k+1):
            fac=fac*(2*j+1)/(2*j)
        fac=fac*math.pi/2
    return fac

def save_image(trajectory,name,nrow=25):
    grid = make_grid(trajectory,nrow=nrow,padding=1,pad_value=.5)
    plt.imsave(name,torch.clip(grid.permute(1,2,0),0,1).cpu().numpy())
    return

class cut_patches_periodic_padding(torch.nn.Module):
    def __init__(self,img_height,img_width,channels,patch_size):
        super(cut_patches_periodic_padding,self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.patch_size = patch_size
        
        self.patch_width=torch.zeros((channels,patch_size,patch_size),dtype=torch.long,device=device)
        self.patch_width+=torch.arange(patch_size,device=device)[None,None,:]
        self.patch_height=torch.zeros((channels,patch_size,patch_size),dtype=torch.long,device=device)
        self.patch_height+=torch.arange(patch_size,device=device)[None,:,None]
        
    def forward(self,imgs,position_inds_height,position_inds_width):
        N=imgs.shape[0]
        n_projections=position_inds_height.shape[0]
        patches_width=(self.patch_width[None,:,:,:].tile(n_projections,1,1,1)+position_inds_width[:,None,None,None])%self.img_width
        patches_height=(self.patch_height[None,:,:,:].tile(n_projections,1,1,1)+position_inds_height[:,None,None,None])%self.img_height
        linear_inds=patches_width+self.img_width*patches_height+(self.img_width*self.img_height)*torch.arange(self.channels,device=device)[None,:,None,None]
        linear_inds=linear_inds.reshape(n_projections,1,-1).tile(1,N,1)
        linear_inds+=(self.channels*self.img_height*self.img_width)*torch.arange(N,device=device)[None,:,None]
        linear_inds=linear_inds.reshape(-1)
        patches=imgs.reshape(-1)[linear_inds].reshape(n_projections,N,self.channels,self.patch_size,self.patch_size)
        return patches,linear_inds
