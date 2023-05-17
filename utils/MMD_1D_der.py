#!/usr/bin/env python

# This code belongs to the paper
#
# J. Hertrich, C. Wald, F. Altekrüger and P. Hagemann (2023)
# Generative Sliced MMD Flows with Riesz Kernels.
# Arxiv Preprint ...
# 

import torch
import math
import numpy as np

device='cuda'
dtype=torch.float


def MMD_derivative_1d(x,y):
    # Compute the derivative of the 1D MMD for the empirical measures encoded in x and y
    # The function takes a batched vectors x and y of size [batch_size,number_of_points]

    N=x.shape[1]
    P=1
    if len(x.shape)>1:
        P=x.shape[0]
    # potential energy
    M=y.shape[1]
    _,inds=torch.sort(torch.cat((x,y),1))
    grad=torch.where(inds>=N,1.,0.).type(dtype)
    grad=(2*torch.cumsum(grad,-1)-M) / (N*M)
    _,inverted_inds=torch.sort(inds)
    inverted_inds=inverted_inds[:,:N]+torch.arange(P,device=device).unsqueeze(1)*(N+M)
    inverted_inds=torch.flatten(inverted_inds)
    grad=grad.flatten()
    grad=grad[inverted_inds].reshape(P,-1)

    # interaction energy
    _,inds_x=torch.sort(x)
    inds_x=inds_x+torch.arange(P,device=device).unsqueeze(1)*N
    inds_x=torch.flatten(inds_x)
    interaction=2*torch.arange(N,dtype=dtype,device=device)-N+1
    interaction=(1/(N**2)) * interaction
    interaction=interaction.tile(P,1)
    grad=grad.flatten()
    grad[inds_x]=grad[inds_x]-interaction.flatten()
    grad=grad.reshape(P,-1)

    return grad


def sliced_factor(d):
    # Compute the factor c_d=c_{d,1}
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


