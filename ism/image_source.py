import numpy as np
import torch
from torch import fft

from defaults import *

# RIR generator for room simulation
# Torch tensor arguments (gradient computation available):
# room_dims : rectangular room dimensions with shape (3, ), representing length, width, height of room
#             coordinate system is such that the room is centered at the origin with length, width, height axes aligned to x, y, z
#             i.e. the walls are the planes (+- l/2, *, *), (*, +- w/2, *), (*, *, +- h/2)
# room_mat : wall absorption with shape (3, 2), with the value at index (i, j) representing the absorption coefficient
#            of the wall aligned to the x (i=0), y (i=1), or z (i=2) axis on the negative (j=0) or positive (j=1) side
# src_loc : source location with shape (3, ), representing x, y, z coords
# mic_loc : microphone location with shape (3, ), representing x, y, z coords
# Other parameters:
# order : order of reflections to compute
# srate : sample rate
# rir_len : length of impulse response to output
# mach : speed of sound
# attenuation : acoustic attenuation constant in air, modeled as per https://ccrma.stanford.edu/~jos/pasp/Air_Absorption.html
# normalize : when set to True, chop off the part of the rir before the direct path
# eval: when set to True, generates unit impulses for the sake of rendering audio, otherwise uses a half-gaussian approximation for smoother gradients
# sigma : float for width of gaussians to represent impulse function (in seconds), setting this value to 0 is equivalent to providing eval=True
def get_rir(room_dims, room_mat, src_loc, mic_loc, order=REF_ORDER, srate=SRATE, rir_len=RIR_LEN, mach=MACH, attenuation=ATTENUATION,
            normalize=False, eval=False, sigma=IMPULSE_SIGMA):
    # construct RIR as a sum of delayed and scaled impulses
    if (eval or abs(sigma) <= 0.000001):
        # sinc-interpolated unit impulse
        impulse_func = lambda t : torch.sinc(t * srate)
    else:
        # half-gaussian (for relaxed optimization)
        impulse_func = lambda t : torch.exp(-1.0 * torch.square(t / sigma)) * (t >= 0.0)
    timestamp = torch.arange(rir_len) / srate
    if (normalize):
        delay_offset = -1.0 * torch.sqrt(torch.sum(torch.square(src_loc - mic_loc))) / mach
    else:
        delay_offset = 0.0
    # precompute lattice values (coordinates, reflectivity)
    # shape of (order, 3, 2) for order max reflections, 3 dimensions, and 2 directions (negative, positive); computed independently in each dim
    lattice_coords = torch.empty((order + 1, 3, 2), requires_grad=False)
    lattice_coords[0,:,:] = src_loc.clone().reshape((3,1)).expand(-1,2)
    lattice_ref = torch.empty((order + 1, 3, 2), requires_grad = False)
    lattice_ref[0,:,:] = torch.ones((3, 2))
    lattice_d = 2.0 * torch.stack([0.5 * room_dims + src_loc, 0.5 * room_dims - src_loc], dim=0) # per-dimension inter-cell lengths in the lattice
    for order_i in range(1, order + 1):
        order_parity = order_i % 2
        lattice_coords[order_i,:,0] = lattice_coords[order_i-1,:,0] - lattice_d[1 - order_parity,:]
        lattice_coords[order_i,:,1] = lattice_coords[order_i-1,:,1] + lattice_d[order_parity,:]
        lattice_ref[order_i,:,0] = -1.0 * lattice_ref[order_i-1,:,0] * room_mat[:,1 - order_parity]
        lattice_ref[order_i,:,1] = -1.0 * lattice_ref[order_i-1,:,1] * room_mat[:,order_parity]
    # quadrant bits, for convenience in traversing lattice
    quadrant_b = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.intc)
    # compute image sources
    rir = torch.zeros((rir_len,))
    for l_i in range(order + 1):
        for l_j in range(order - l_i + 1):
            for l_k in range(order - l_i - l_j + 1):
                # per-quadrant de-duplication mask: don't recompute lattice points along boundaries
                l_ijk = np.array([l_i, l_j, l_k], dtype=np.intc)
                quadrant_mask = np.all(np.logical_not((l_ijk[:,np.newaxis] == 0) & (quadrant_b !=0)), axis=0)
                for quad_i in range(8):
                    quad_bi = quadrant_b[:,quad_i]
                    if quadrant_mask[quad_i]:
                        # not sure if torch has np style index slicing, so for now this will do
                        im_loc = torch.stack([lattice_coords[l_i, 0, quad_bi[0]],
                                              lattice_coords[l_j, 1, quad_bi[1]],
                                              lattice_coords[l_k, 2, quad_bi[2]]], dim=0)
                        im_ref = lattice_ref[l_i, 0, quad_bi[0]] * lattice_ref[l_j, 1, quad_bi[1]] * lattice_ref[l_k, 2, quad_bi[2]]
                        delay, atten = compute_path_t(im_loc, mic_loc, attenuation)
                        rir = rir + im_ref * atten * impulse_func(timestamp - (delay + delay_offset))
    return rir

# Convenience function for computing the delay and attenuation corresponding to a given sound path
def compute_path_t(src_loc, mic_loc, attenuation):
    dist = torch.sqrt(torch.sum(torch.square(src_loc - mic_loc)))
    delay = dist / MACH
    atten = torch.exp(-1.0 * attenuation * dist)
    return delay, atten
