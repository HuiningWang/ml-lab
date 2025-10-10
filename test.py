import torch
import numpy as np
from torch import nn


class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()