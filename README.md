# Generative Sliced MMD Flows with Riesz Kernels

Implementation to the paper "Generative Sliced MMD Flows with Riesz Kernels" available at https://arxiv.org/abs/2305.11463. 

The files `run_MNIST.py`, `run_FashionMNIST.py` and `run_CIFAR.py` reproduce the training of generative sliced MMD flows from Section 5 of the paper.
The code is written in PyTorch 2.0.0.

If you have any questions, feel free to contact  
Johannes Hertrich (j.hertrich@math.tu-berlin.de)  
Christian Wald (wald@math.tu-berlin.de)  
Fabian Altekrüger (fabian.altekrueger@hu-berlin.de)  
or Paul Hagemann (hagemann@math.tu-berlin.de).

## USAGE 

You can start the training of the generative sliced MMD flow by calling the script 'run_{dataset}.py' for the datasets MNIST, FashionMNIST and CIFAR10. If you do not wish to save intermediate steps of the flow, then set the flag 'save' to False.
If you want to generate new samples using the proposed scheme, then set the flag 'visualize' to True.

## REFERENCE

[1] J. Hertrich, C. Wald, F. Altekrüger and P. Hagemann (2023)  
Generative Sliced MMD Flows with Riesz Kernels.  
Preprint available under https://arxiv.org/abs/2305.11463
