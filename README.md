# Generative Sliced MMD Flows with Riesz Kernels

Implementation to the paper "Generative Sliced MMD Flows with Riesz Kernels" available at https://openreview.net/forum?id=VdkGRV1vcf. 

The files `run_MNIST.py`, `run_FashionMNIST.py`, `run_CIFAR.py` and `run_celebA.py` reproduce the training of generative sliced MMD flows from Section 5 of the paper.
The code is written in PyTorch 2.0.0.

If you have any questions, feel free to contact Johannes Hertrich (j.hertrich@math.tu-berlin.de), Christian Wald (wald@math.tu-berlin.de), Fabian Altekrüger (fabian.altekrueger@hu-berlin.de) 
or Paul Hagemann (hagemann@math.tu-berlin.de).

## USAGE 

You can start the training of the generative sliced MMD flow by calling the script 'run_{dataset}.py' for the datasets MNIST, FashionMNIST, CIFAR10 and CelebA. As an example, if you want to start the training for MNIST, you need to call
```python
python run_MNIST.py
```
If you do not wish to save intermediate steps of the flow, then set the flag 'save' to False.
If you want to generate new samples using the proposed scheme, then set the flag 'visualize' to True.
```python
python run_MNIST.py --visualize=True
```

## REFERENCE

[1] J. Hertrich, C. Wald, F. Altekrüger and P. Hagemann (2024)  
Generative Sliced MMD Flows with Riesz Kernels.  
International Conference on Learning Representations.

## CITATION
```python
@inproceedings{HWAH2024,
    author    = {Hertrich, Johannes and Wald, Christan and Altekrüger, Fabian and Hagemann, Paul},
    title     = {Generative Sliced {MMD} Flows with {Riesz} Kernels},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024}
}
```
