import numpy as np
import torch
from torch import autograd
from torch import fft

from defaults import *

# RIR generator for room simulation
# Torch tensor arguments (gradient computation available; all arguments can be batched so long as the batch shapes are broadcastable with each other):
# room_dims : rectangular room dimensions with shape ending in (3, ), representing length, width, height of room
#             coordinate system is such that the room is centered at the origin with length, width, height axes aligned to x, y, z
#             i.e. the walls are the planes (+- l/2, *, *), (*, +- w/2, *), (*, *, +- h/2)
# room_mat : wall absorption with shape ending in (3, 2), with the value at index (i, j) representing the absorption coefficient
#            of the wall aligned to the x (i=0), y (i=1), or z (i=2) axis on the negative (j=0) or positive (j=1) side
# src_loc : source location(s) with shape ending in (3, ), representing x, y, z coords
# mic_loc : microphone location(s) with shape ending in (3, ), representing x, y, z coords
# Other parameters:
# order : order of reflections to compute via ISM
# srate : sample rate
# rir_len : length of impulse response to output
# mach : speed of sound
# attenuation : acoustic attenuation constant in air, modeled as per https://ccrma.stanford.edu/~jos/pasp/Air_Absorption.html
# normalize : when set to True, chop off the part of the rir before the direct path
# normalize_mode : when set to 'batch', will preserve the relative direct path delays between output RIRs
#                  i.e. will chop off only the amount before the earliest direct path arrival;
#                  otherwise will normalize on a per-sample basis
# eval: when set to True, generates unit impulses for the sake of rendering audio, otherwise uses a half-gaussian approximation for smoother gradients
# render_tail : when set to True, render an approximate tail
# sigma : float for width of gaussians to represent impulse function (in seconds), setting this value to 0 is equivalent to providing eval=True
def get_rir(room_dims, room_mat, src_loc, mic_loc, order=REF_ORDER, srate=SRATE, rir_len=RIR_LEN, mach=MACH, attenuation=ATTENUATION,
            normalize=False, normalize_mode='each', eval=False, render_tail=True, sigma=IMPULSE_SIGMA):
    batch_dims = max(room_dims.shape[:-1], room_mat.shape[:-2], src_loc.shape[:-1], mic_loc.shape[:-1])
    # construct RIR as a sum of delayed and scaled impulses
    if (eval or abs(sigma) <= 0.000001):
        # sinc-interpolated unit impulse
        impulse_func = lambda t : torch.sinc(t * srate)
    else:
        # half-gaussian (for relaxed optimization)
        impulse_func = lambda t : torch.exp(-1.0 * torch.square(t / sigma)) * (t >= 0.0)
    timestamp = torch.arange(rir_len) / srate
    timestamp = timestamp.view((1,)*len(batch_dims) + timestamp.shape)
    direct_delay = torch.sqrt(torch.sum(torch.square(src_loc - mic_loc), dim=-1, keepdim=True)) / mach
    if (normalize):
        if (normalize_mode == 'batch'):
            delay_offset = -1.0 * torch.min(direct_delay)
        else:
            delay_offset = -1.0 * direct_delay
    else:
        delay_offset = 0.0
    # precompute lattice values (coordinates, reflectivity)
    # shape of (order, 3, 2) for order max reflections, 3 dimensions, and 2 directions (negative, positive); computed independently in each dim
    lattice_coords = torch.empty(batch_dims + (order + 1, 3, 2), requires_grad=False)
    lattice_coords[...,0,:,:] = torch.unsqueeze(src_loc.clone(), dim=-1).expand(src_loc.shape + (2,))
    lattice_ref = torch.empty(batch_dims + (order + 1, 3, 2), requires_grad=False)
    lattice_ref[...,0,:,:] = torch.ones(batch_dims + (3, 2))
    lattice_d = 2.0 * torch.stack([0.5 * room_dims + src_loc, 0.5 * room_dims - src_loc], dim=-2) # per-dimension inter-cell lengths in the lattice
    for order_i in range(1, order + 1):
        order_parity = order_i % 2
        lattice_coords[...,order_i,:,0] = lattice_coords[...,order_i-1,:,0] - lattice_d[...,1 - order_parity,:]
        lattice_coords[...,order_i,:,1] = lattice_coords[...,order_i-1,:,1] + lattice_d[...,order_parity,:]
        lattice_ref[...,order_i,:,0] = -1.0 * lattice_ref[...,order_i-1,:,0] * room_mat[...,:,1 - order_parity]
        lattice_ref[...,order_i,:,1] = -1.0 * lattice_ref[...,order_i-1,:,1] * room_mat[...,:,order_parity]
    # quadrant bits, for convenience in traversing lattice
    quadrant_b = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.intc)
    # compute image sources
    rir = torch.zeros(batch_dims + (rir_len,))
    if (render_tail):
        max_order_amp = []
        max_order_delay = []
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
                        im_loc = torch.stack([lattice_coords[..., l_i, 0, quad_bi[0]],
                                              lattice_coords[..., l_j, 1, quad_bi[1]],
                                              lattice_coords[..., l_k, 2, quad_bi[2]]], dim=-1)
                        im_ref = lattice_ref[..., l_i, 0, quad_bi[0]] * lattice_ref[..., l_j, 1, quad_bi[1]] * lattice_ref[..., l_k, 2, quad_bi[2]]
                        delay, atten = compute_path_t(im_loc, mic_loc, attenuation)
                        if (len(batch_dims) > 0):
                            im_ref = im_ref[...,None]
                            atten = atten[...,None]
                            delay = delay[...,None]
                        if (render_tail and (l_i + l_j + l_k == order)):
                            max_order_amp.append(im_ref * atten)
                            max_order_delay.append(delay - direct_delay)
                        rir = rir + im_ref * atten * impulse_func(timestamp - (delay + delay_offset))
    # render approximate tail
    if (render_tail):
        # generate enveloped noise tails based on last image sources
        tail_func = lambda t, d, t0 : torch.exp(d * t) * (t > t0)
        tail_offset = 0.0 if normalize else (-1.0 * direct_delay)
        max_order_amp = torch.stack(max_order_amp, dim=-1)
        max_order_delay = torch.stack(max_order_delay, dim=-1)
        max_order_decay = torch.log(torch.abs(max_order_amp) + torch.finfo(torch.float32).eps * 1000) / max_order_delay
        tail_env = tail_func((timestamp + tail_offset)[...,None], max_order_decay, max_order_delay * (order + 1.0) / order)
        tail = torch.sum(0.5 * torch.randn(timestamp.shape + tail_env.shape[-1:]) * tail_env / max(6.0, order), dim=-1)
        rir = rir + tail
    return rir

# Convenience function for computing the delay and attenuation corresponding to a given sound path
def compute_path_t(src_loc, mic_loc, attenuation):
    dist = torch.sqrt(torch.sum(torch.square(src_loc - mic_loc), dim=-1))
    delay = dist / MACH
    atten = torch.exp(-1.0 * attenuation * dist)
    return delay, atten

# RIR generator for room simulation with smoothed gradients
# Torch tensor arguments (gradient computation available, arguments can be batched):
# input : tensor with shape ending in (15, ), consisting of, in order, the arguments room_dims, room_mat, src_loc, mic_loc with the non-batch dimensions flattened:
#    room_dims : rectangular room dimensions with shape ending in (3, ), representing length, width, height of room
#                coordinate system is such that the room is centered at the origin with length, width, height axes aligned to x, y, z
#                i.e. the walls are the planes (+- l/2, *, *), (*, +- w/2, *), (*, *, +- h/2)
#    room_mat  : wall absorption with shape ending in (3, 2), with the value at index (i, j) representing the absorption coefficient
#                of the wall aligned to the x (i=0), y (i=1), or z (i=2) axis on the negative (j=0) or positive (j=1) side
#    src_loc   : source location(s) with shape ending in (3, ), representing x, y, z coords
#    mic_loc   : microphone location(s) with shape ending in (3, ), representing x, y, z coords
# Other arguments:
# options: dict consisting of the following options that can be specified
#    order          : order of reflections to compute via ISM
#    srate          : sample rate
#    rir_len        : length of impulse response to output
#    mach           : speed of sound
#    attenuation    : acoustic attenuation constant in air, modeled as per https://ccrma.stanford.edu/~jos/pasp/Air_Absorption.html
#    normalize      : when set to True, chop off the part of the rir before the direct path
#    normalize_mode : when set to 'batch', will preserve the relative direct path delays between output RIRs
#    render_tail    : when set to True, render an approximate tail
#    sigma          : float for width of gaussians to represent impulse function (in seconds) during gradient computation
class SmoothGradISM(autograd.Function):
    @staticmethod
    def forward(ctx, input, options):
        # process options and defaults
        order = options['order'] if ('order' in options) else REF_ORDER
        srate = options['srate'] if ('srate' in options) else SRATE
        rir_len = options['rir_len'] if ('rir_len' in options) else RIR_LEN
        mach = options['mach'] if ('mach' in options) else MACH
        attenuation = options['attenuation'] if ('attenuation' in options) else ATTENUATION
        normalize = options['normalize'] if ('normalize' in options) else False
        normalize_mode = options['normalize_mode'] if ('normalize_mode' in options) else 'each'
        render_tail = options['render_tail'] if ('render_tail' in options) else True
        sigma = options['sigma'] if ('sigma' in options) else IMPULSE_SIGMA
        with torch.enable_grad():
            # input bookkeeping
            batch_dims = input.shape[:-1]
            room_dims = input[...,:3]
            room_mat = input[...,3:9,None].reshape(batch_dims + (3,2))
            src_loc = input[...,9:12]
            mic_loc = input[...,12:15]
            # define impulse functions: regular sinc-interpolated delta plus a gaussian approximation for gradient computation
            impulse_func = lambda t : torch.sinc(t * srate)
            impulse_func_g = lambda t : torch.exp(-1.0 * torch.square(t / sigma))
            timestamp = torch.arange(rir_len) / srate
            timestamp = timestamp.view((1,)*len(batch_dims) + timestamp.shape)
            # compute delay compensation for direct path
            direct_delay = torch.sqrt(torch.sum(torch.square(src_loc - mic_loc), dim=-1, keepdim=True)) / mach
            if (normalize):
                if (normalize_mode == 'batch'):
                    delay_offset = -1.0 * torch.min(direct_delay)
                else:
                    delay_offset = -1.0 * direct_delay
            else:
                delay_offset = 0.0
            # precompute lattice values (coordinates, reflectivity)
            # shape of (order, 3, 2) for order max reflections, 3 dimensions, and 2 directions (negative, positive); computed independently in each dim
            lattice_coords = torch.empty(batch_dims + (order + 1, 3, 2), requires_grad=False)
            lattice_coords[...,0,:,:] = torch.unsqueeze(src_loc.clone(), dim=-1).expand(src_loc.shape + (2,))
            lattice_ref = torch.empty(batch_dims + (order + 1, 3, 2), requires_grad=False)
            lattice_ref[...,0,:,:] = torch.ones(batch_dims + (3, 2))
            lattice_d = 2.0 * torch.stack([0.5 * room_dims + src_loc, 0.5 * room_dims - src_loc], dim=-2) # per-dimension inter-cell lengths in the lattice
            for order_i in range(1, order + 1):
                order_parity = order_i % 2
                lattice_coords[...,order_i,:,0] = lattice_coords[...,order_i-1,:,0] - lattice_d[...,1 - order_parity,:]
                lattice_coords[...,order_i,:,1] = lattice_coords[...,order_i-1,:,1] + lattice_d[...,order_parity,:]
                lattice_ref[...,order_i,:,0] = -1.0 * lattice_ref[...,order_i-1,:,0] * room_mat[...,:,1 - order_parity]
                lattice_ref[...,order_i,:,1] = -1.0 * lattice_ref[...,order_i-1,:,1] * room_mat[...,:,order_parity]
            # quadrant bits, for convenience in traversing lattice
            quadrant_b = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                                [0, 0, 1, 1, 0, 0, 1, 1],
                                [0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.intc)
            # compute image sources
            rir = torch.zeros(batch_dims + (rir_len,), requires_grad=False)
            rir_g = torch.zeros_like(rir, requires_grad=True)
            if (render_tail):
                max_order_amp = []
                max_order_delay = []
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
                                im_loc = torch.stack([lattice_coords[..., l_i, 0, quad_bi[0]],
                                                    lattice_coords[..., l_j, 1, quad_bi[1]],
                                                    lattice_coords[..., l_k, 2, quad_bi[2]]], dim=-1)
                                im_ref = lattice_ref[..., l_i, 0, quad_bi[0]] * lattice_ref[..., l_j, 1, quad_bi[1]] * lattice_ref[..., l_k, 2, quad_bi[2]]
                                #delay, atten = compute_path_t(im_loc, mic_loc, attenuation)
                                im_dist = torch.sqrt(torch.sum(torch.square(im_loc - mic_loc), dim=-1))
                                delay = im_dist / MACH
                                atten = torch.exp(-1.0 * attenuation * im_dist)
                                if (len(batch_dims) > 0):
                                    im_ref = im_ref[...,None]
                                    atten = atten[...,None]
                                    delay = delay[...,None]
                                if (render_tail and (l_i + l_j + l_k == order)):
                                    max_order_amp.append(im_ref * atten)
                                    max_order_delay.append(delay - direct_delay)
                                rir = rir + im_ref * atten * impulse_func(timestamp - (delay + delay_offset))
                                rir_g = rir_g + im_ref * atten * impulse_func_g(timestamp - (delay + delay_offset))
            # render approximate tail
            if (render_tail):
                # generate enveloped noise tails based on last image sources
                tail_func = lambda t, d, t0 : torch.exp(d * t) * (t > t0)
                tail_offset = 0.0 if normalize else (-1.0 * direct_delay)
                max_order_amp = torch.stack(max_order_amp, dim=-1)
                max_order_delay = torch.stack(max_order_delay, dim=-1)
                max_order_decay = torch.log(torch.abs(max_order_amp) + torch.finfo(torch.float32).eps * 1000) / max_order_delay
                tail_env = tail_func((timestamp + tail_offset)[...,None], max_order_decay, max_order_delay * (order + 1.0) / order)
                tail = torch.sum(0.5 * torch.randn(timestamp.shape + tail_env.shape[-1:]) * tail_env / max(6.0, order), dim=-1)
                rir = rir + tail
                rir_g = rir_g + tail
        ctx.save_for_backward(input, rir_g)
        return rir
    @staticmethod
    def backward(ctx, grad_output):
        input, rir_g = ctx.saved_tensors
        with torch.enable_grad():
            grad_g = autograd.grad(rir_g, input, grad_output)
        if (len(grad_g) > 1):
            grad_g = torch.stack(grad_g, dim=0)
        else:
            grad_g = grad_g[0]
        return grad_g, None
