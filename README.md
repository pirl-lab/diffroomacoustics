# diffroomacoustics
A Differentiable Room Acoustics Simulator written in PyTorch

Currently implemented models:
- Image Source Method
    - Efficient implementation based on the [fast image method by McGovern](https://www.sciencedirect.com/science/article/abs/pii/S0003682X08000455)

## Simple Usage Example (sets up and solves an optimization problem for RT60):
```python
import torch
import numpy as np
import matplotlib.pyplot as plt

from ism.image_source import get_rir
from metrics.metrics import reverb_time

# Ground Truth Scene
room_dims_gt = torch.tensor([35.0, 25.0, 10.0], requires_grad=False)
room_mat_gt = torch.tensor([[0.5, 0.5],
                         [0.5, 0.4],
                         [0.6, 0.2]], requires_grad=False)
src_loc_gt = torch.tensor([-15.0, 10.0, 5.0], requires_grad=False)
mic_loc_gt = torch.tensor([1.5, -1.0, 2.0], requires_grad=False)
rir_gt = get_rir(room_dims_gt, room_mat_gt, src_loc_gt, mic_loc_gt, order=5, normalize=True)

# Optimize for room material via Adam
room_dims = torch.tensor([35.0, 25.0, 10.0], requires_grad=True)
room_mat = torch.tensor([[0.75, 0.25],
                         [0.5, 0.4],
                         [0.1, 0.9]], requires_grad=True)
src_loc = torch.tensor([-15.0, 10.0, 5.0], requires_grad=True)
mic_loc = torch.tensor([1.5, -1.0, 2.0], requires_grad=True)

NUM_ITER = 50 # number of optimzer iterations

opt = torch.optim.Adam([room_mat], lr=0.1)

history = {'iter' : [], 'rir_error' : [], 'room_mat' : []}
for iter_i in range(NUM_ITER):
    opt.zero_grad()
    rir_guess = get_rir(room_dims, room_mat, src_loc, mic_loc, order=5, normalize=True)
    rir_err = torch.abs(reverb_time(rir_gt) - reverb_time(rir_guess))
    history['iter'].append(iter_i)
    history['rir_error'].append(rir_err.detach().clone().numpy())
    history['room_mat'].append(room_mat.detach().clone().numpy())
    rir_err.backward()
    opt.step()

history['room_mat'] = np.array(history['room_mat'])

# Plot error curve
plt.plot(history['iter'], history['rir_error'])
plt.xlabel('iteration')
plt.ylabel('RT60 error')
```
