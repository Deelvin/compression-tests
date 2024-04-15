from enum import Enum

import numpy as np
import torch

class ClippingStrategy(Enum):
    KL = 1
    Percentile = 2
    MSE = 3
