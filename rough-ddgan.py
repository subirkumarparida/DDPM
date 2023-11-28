# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:21:02 2023

@author: subir
"""
import numpy as np
import torch
import torch.distributed as dist

# def broadcast_params(params):
#     for param in params:
#         dist.broadcast(param.data, src=0)
        
            
# foo = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(foo)
# dist.broadcast(foo, 0)

# t = torch.tensor([[5, 2], [3, 4]])
# print(torch.gather(t, 0, torch.tensor([[1, 1], [1, 1]])))

# print(np.arange(0, 10 + 1, dtype=np.float64))

# def get_time_schedule():
#     n_timestep = 10
#     eps_small = 1e-3
#     t = np.arange(0, n_timestep + 1, dtype=np.float64)
#     print(t)
#     t = t / n_timestep
#     print(t)
#     t = torch.from_numpy(t) * (1. - eps_small) + eps_small
#     print(t)
#     print(t.shape)
#     return t

# get_time_schedule()

first = torch.tensor(1e-8)
test = torch.tensor([1, 4, 3])
print(first[None])

betas = torch.cat((first[None], test))
print(betas)
betas = betas.type(torch.float32)
print(betas)
