import numpy as np
import random

import torch

from Dataset import *
from Agents import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from GuessAction import *
import matplotlib.pyplot as plt

def plotfig(fp):
    x=np.arange(0,20000,50 )
    y=[]
    with open(fp,'r') as f:
        data=f.read().strip().split()
        y.extend([float(i) for i in data])
    plt.plot(x,y)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("EVAL")
    plt.show()
    plt.savefig("pictures/gr_accte_ConvNet.png")

if __name__ == "__main__":
    plotfig("results/gr_accte_ConvNet.txt")