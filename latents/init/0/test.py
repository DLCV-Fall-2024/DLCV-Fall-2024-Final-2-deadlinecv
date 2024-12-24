import torch
import os
code_dir = os.path.dirname(os.path.abspath(__file__))
ten1 = torch.load(os.path.join(code_dir, "0.pt"))
ten2 = torch.load(os.path.join(code_dir, "0 (1).pt"))

# check if tensors are the same
print(torch.allclose(ten1, ten2)) # True