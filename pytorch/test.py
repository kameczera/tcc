from typing import List
import torch
import torchdynamo

torchdynamo.config.debug = True

def toy_example(a, b):
  return a + b

with torchdynamo.optimize("eager"):
  toy_example(torch.randn(10), torch.randn(10))