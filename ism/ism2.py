import numpy as np
from scipy.signal import iirfilter
import torch
from torch import fft
from torchaudio.functional import lfilter

from defaults import *

# RIR generator for room simulation (updated version)
# Torch tensor arguments (gradient computation available; all arguments can be batched so long as the batch shapes are broadcastable with each other):
# room_dims : rectangular room dimensions with shape ending in (3, ), representing length, width, height of room
#             coordinate system is such that the room is centered at the origin with length, width, height axes aligned to x, y, z
#             i.e. the walls are the planes (+- l/2, *, *), (*, +- w/2, *), (*, *, +- h/2)
# room_mat : wall reflectance with shape ending in (3, 2) or (3, 2, L), with the value at index (i, j) representing the (L frequency-dependent) reflectance coefficient
#            of the wall aligned to the x (i=0), y (i=1), or z (i=2) axis on the negative (j=0) or positive (j=1) side
# src_loc : source location(s) with shape ending in (3, ), representing x, y, z coords
# mic_loc : microphone location(s) with shape ending in (3, ), representing x, y, z coords
# Other parameters:
# order : order of reflections to compute via ISM
# srate : sample rate
# rir_len : length of impulse response to output
# mat_freqs : ascending frequencies with shape (L, ) corresponding to the last dimension of room_mat (if given)
#             uses the linear frequencies represented by the even 2 * (L - 1) length DFT by default
# wall_filt_len : when mat_freqs is provided, the wall filter is computed via linear interpolation to DFT frequencies of length wall_filt_len
#                 if greater than 0.1 * rir_len is provided, the latter is used instead
# mach : speed of sound
# attenuation : acoustic attenuation constant in air, modeled as per https://ccrma.stanford.edu/~jos/pasp/Air_Absorption.html
# normalize : when set to True, chop off the part of the rir before the direct path
# normalize_mode : when set to 'batch', will preserve the relative direct path delays between output RIRs
#                  i.e. will chop off only the amount before the earliest direct path arrival;
#                  otherwise will normalize on a per-sample basis
# image_sigma : when set to a positive value, image source positions will be perturbed via a zero-mean uniform distribution with image_sigma range in all dimensions
# hpf : when true, use a 10 Hz highpass filter to remove DC bias from RIR
# render_tail : when set to True, render an approximate tail
def get_rir(room_dims, room_mat, src_loc, mic_loc, order=REF_ORDER, srate=SRATE, rir_len=RIR_LEN, mat_freqs=None, wall_filt_len=1024,
            mach=MACH, attenuation=ATTENUATION, normalize=False, normalize_mode='each', image_sigma=0, hpf=True, render_tail=True):
    # shape check room_mat
    assert ((room_mat.shape[-2:] == (3, 2)) or (room_mat.shape[-3:-1] == (3, 2))), "invalid room_mat shape"
    multifreq = (room_mat.shape[-3:-1] == (3, 2))
    # construct RIR as a sum of delayed and scaled impulse responses
    rfft_len = rir_len // 2 + 1
    impulse = torch.ones(rfft_len, requires_grad=False, device=room_dims.device)
    if (multifreq):
        if (room_mat.shape[-1] > 1):
            # compute min-phase wall filters
            if (mat_freqs is None):
                mat_freqs = torch.arange(room_mat.shape[-1], requires_grad=False, device=room_mat.device) * srate / (2 * (room_mat.shape[-1] - 1))
            wall_f_len = min(int(0.1 * rir_len), wall_filt_len)
            wall_rfft_len = wall_f_len // 2 + 1
            dft_freqs = torch.arange(wall_rfft_len, requires_grad=False, device=room_mat.device) * srate / wall_f_len
            dft_freqs = dft_freqs.reshape((1,)*len(room_mat.shape[:-1]) + dft_freqs.shape).expand(room_mat.shape[:-1] + (-1,))
            mat_freqs = mat_freqs.reshape((1,)*len(room_mat.shape[:-1]) + mat_freqs.shape[-1:]).expand(room_mat.shape[:-1] + (-1,))
            room_mat_ = interp(dft_freqs, mat_freqs, room_mat, dim=-1)
            room_mat_ = minphase(room_mat_, wall_f_len)
            if (wall_f_len != rir_len):
                room_mat_ = fft.rfft(fft.irfft(room_mat_, n=wall_f_len, norm='backward'), n=rir_len, norm='backward')
    else:
        wall_f_len = 1
        room_mat_ = room_mat[...,None]
    max_delay = (rir_len - wall_f_len - 1) / srate # prevent circular shifting
    if (render_tail):
        timestamp = torch.arange(rir_len, requires_grad=False, device=room_dims.device) / srate
    omega = (-2.0j * torch.pi * srate / rir_len) * torch.arange(rfft_len, requires_grad=False, device=room_dims.device)
    direct_dist = torch.sqrt(torch.sum(torch.square(src_loc - mic_loc), dim=-1, keepdim=True))
    if (normalize):
        if (normalize_mode == 'batch'):
            delay_offset = -1.0 * torch.min(direct_dist / mach)
            dist_norm = torch.min(direct_dist)
        else:
            delay_offset = -1.0 * direct_dist / mach
            dist_norm = direct_dist
    else:
        delay_offset = 0.0
        dist_norm = direct_dist
    # precompute lattice values (coordinates, reflectivity)
    # shape of (order, 3, 2) for order max reflections, 3 dimensions, and 2 directions (negative, positive); computed independently in each dim
    lattice_d = 2.0 * torch.stack([0.5 * room_dims + src_loc, 0.5 * room_dims - src_loc], dim=-1) # per-dimension inter-cell lengths in the lattice
    lattice_coords = torch.empty(lattice_d.shape[:-2] + (order + 1, 3, 2), requires_grad=False, device=lattice_d.device)
    lattice_coords[...,0,:,:] = torch.unsqueeze(src_loc.clone(), dim=-1).expand(src_loc.shape + (2,))
    lattice_ref = torch.ones(room_mat_.shape[:-3] + (order + 1, 3, 2, room_mat_.shape[-1]), requires_grad=False, device=room_mat_.device, dtype=room_mat_.dtype)
    for order_i in range(1, order + 1):
        order_parity = order_i % 2
        lattice_coords[...,order_i,:,0] = lattice_coords[...,order_i-1,:,0] - lattice_d[...,:,1 - order_parity]
        lattice_coords[...,order_i,:,1] = lattice_coords[...,order_i-1,:,1] + lattice_d[...,:,order_parity]
        lattice_ref[...,order_i,:,0,:] = 1.0 * lattice_ref[...,order_i-1,:,0,:] * room_mat_[...,:,1 - order_parity,:]
        lattice_ref[...,order_i,:,1,:] = 1.0 * lattice_ref[...,order_i-1,:,1,:] * room_mat_[...,:,order_parity,:]
    # quadrant bits, for convenience in traversing lattice
    quadrant_b = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.intc)
    # compute image sources
    rir = 0.0
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
                        if (image_sigma > 0) and (l_i + l_j + l_k > 0):
                            im_loc = im_loc + image_sigma * (torch.rand_like(im_loc) - 0.5)
                        im_ref = (lattice_ref[..., l_i, 0, quad_bi[0],:] * lattice_ref[..., l_j, 1, quad_bi[1],:] * lattice_ref[..., l_k, 2, quad_bi[2],:])
                        delay, atten, dist = compute_path_t(im_loc, mic_loc, mach, attenuation)
                        if (render_tail and (l_i + l_j + l_k == order)):
                            max_order_amp.append((dist_norm / dist) * im_ref * atten)
                            max_order_delay.append(delay + delay_offset)
                        delay_mask = (delay + delay_offset >= 0) * (delay + delay_offset < max_delay)
                        rir = rir + delay_mask * (dist_norm / dist) * im_ref * atten * impulse * torch.exp(omega * (delay + delay_offset))
    rir = fft.irfft(rir, n=rir_len, norm='backward', dim=-1)
    # render approximate tail
    if (render_tail):
        # generate enveloped noise tails based on last image sources
        tail_func = lambda t, d, t0 : torch.exp(d * t) * (t > t0)
        tail_offset = 0.0 if normalize else delay_offset
        max_order_amp = torch.stack(max_order_amp, dim=-1)
        max_order_delay = torch.stack(max_order_delay, dim=-1)
        max_order_decay = safe_log(torch.abs(max_order_amp)) / max_order_delay
        tail_env = tail_func((timestamp + tail_offset)[...,None], max_order_decay, max_order_delay * (order + 1.0) / order)
        tail = torch.sum(0.5 * torch.randn(timestamp.shape + tail_env.shape[-1:], device=timestamp.device) * tail_env / max(6.0, order), dim=-1)
        rir = rir + tail
    if (hpf):
        b, a = iirfilter(2, 10.0, rp=5.0, rs=60.0, btype='highpass', ftype='butter', fs=SRATE, output='ba')
        a = torch.tensor(a, device=rir.device, dtype=rir.dtype)
        b = torch.tensor(b, device=rir.device, dtype=rir.dtype)
        rir = lfilter(rir, a, b)
    return rir

# Precompute image-source lattice coordinates for a set location and shoebox room geometry
# Torch tensor arguments (gradient computation available; all arguments can be batched so long as the batch shapes are broadcastable with each other):
# room_dims : rectangular room dimensions with shape ending in (3, ), representing length, width, height of room
#             coordinate system is such that the room is centered at the origin with length, width, height axes aligned to x, y, z
#             i.e. the walls are the planes (+- l/2, *, *), (*, +- w/2, *), (*, *, +- h/2)
# query_loc : query location(s) with shape ending in (3, ), representing x, y, z coords
# Other parameters:
# order : order of reflections to compute via ISM
def lattice_coord_shoebox(room_dims, query_loc, order=REF_ORDER):
    # precompute lattice values (coordinates, reflectivity)
    # shape of (order, 3, 2) for order max reflections, 3 dimensions, and 2 directions (negative, positive); computed independently in each dim
    lattice_d = 2.0 * torch.stack([0.5 * room_dims + query_loc, 0.5 * room_dims - query_loc], dim=-1) # per-dimension inter-cell lengths in the lattice
    lattice_coords = torch.empty(lattice_d.shape[:-2] + (order + 1, 3, 2), requires_grad=False, device=lattice_d.device)
    lattice_coords[...,0,:,:] = torch.unsqueeze(query_loc.clone(), dim=-1).expand(query_loc.shape + (2,))
    for order_i in range(1, order + 1):
        order_parity = order_i % 2
        lattice_coords[...,order_i,:,0] = lattice_coords[...,order_i-1,:,0] - lattice_d[...,:,1 - order_parity]
        lattice_coords[...,order_i,:,1] = lattice_coords[...,order_i-1,:,1] + lattice_d[...,:,order_parity]
    return lattice_coords

# Precompute image-source material reflection lattice for a shoebox room geometry
# room_mat : wall reflectance with shape ending in (3, 2) or (3, 2, L), with the value at index (i, j) representing the (L frequency-dependent) reflectance coefficient
#            of the wall aligned to the x (i=0), y (i=1), or z (i=2) axis on the negative (j=0) or positive (j=1) side
# Other parameters:
# order : order of reflections to compute via ISM
# srate : sample rate
# rir_len : length of impulse response to output
# mat_freqs : ascending frequencies with shape (L, ) corresponding to the last dimension of room_mat (if given)
#             uses the linear frequencies represented by the even 2 * (L - 1) length DFT by default
# wall_filt_len : when mat_freqs is provided, the wall filter is computed via linear interpolation to DFT frequencies of length wall_filt_len
#                 if greater than 0.1 * rir_len is provided, the latter is used instead
def lattice_ref_shoebox(room_mat, order=REF_ORDER, srate=SRATE, rir_len=RIR_LEN, mat_freqs=None, wall_filt_len=1024):
    # shape check room_mat
    assert ((room_mat.shape[-2:] == (3, 2)) or (room_mat.shape[-3:-1] == (3, 2))), "invalid room_mat shape"
    multifreq = (room_mat.shape[-3:-1] == (3, 2))
    if (multifreq):
        if (room_mat.shape[-1] > 1):
            # compute min-phase wall filters
            if (mat_freqs is None):
                mat_freqs = torch.arange(room_mat.shape[-1], requires_grad=False, device=room_mat.device) * srate / (2 * (room_mat.shape[-1] - 1))
            wall_f_len = min(int(0.1 * rir_len), wall_filt_len)
            wall_rfft_len = wall_f_len // 2 + 1
            dft_freqs = torch.arange(wall_rfft_len, requires_grad=False, device=room_mat.device) * srate / wall_f_len
            dft_freqs = dft_freqs.reshape((1,)*len(room_mat.shape[:-1]) + dft_freqs.shape).expand(room_mat.shape[:-1] + (-1,))
            mat_freqs = mat_freqs.reshape((1,)*len(room_mat.shape[:-1]) + mat_freqs.shape[-1:]).expand(room_mat.shape[:-1] + (-1,))
            room_mat_ = interp(dft_freqs, mat_freqs, room_mat, dim=-1)
            room_mat_ = minphase(room_mat_, wall_f_len)
            if (wall_f_len != rir_len):
                room_mat_ = fft.rfft(fft.irfft(room_mat_, n=wall_f_len, norm='backward'), n=rir_len, norm='backward')
    else:
        wall_f_len = 1
        room_mat_ = room_mat[...,None]
    # precompute lattice values (coordinates, reflectivity)
    # shape of (order, 3, 2) for order max reflections, 3 dimensions, and 2 directions (negative, positive); computed independently in each dim
    lattice_ref = torch.ones(room_mat_.shape[:-3] + (order + 1, 3, 2, room_mat_.shape[-1]), requires_grad=False, device=room_mat_.device, dtype=room_mat_.dtype)
    for order_i in range(1, order + 1):
        order_parity = order_i % 2
        lattice_ref[...,order_i,:,0,:] = 1.0 * lattice_ref[...,order_i-1,:,0,:] * room_mat_[...,:,1 - order_parity,:]
        lattice_ref[...,order_i,:,1,:] = 1.0 * lattice_ref[...,order_i-1,:,1,:] * room_mat_[...,:,order_parity,:]
    return lattice_ref

# RIR generator for room simulation using precomputed lattices
# Torch tensor arguments (gradient computation available; all arguments can be batched so long as the batch shapes are broadcastable with each other):
# lattice_coord : ISM coordinate lattice as generated by lattice_coord_shoebox
# lattice_ref : reflection lattice as generated by lattice_ref_shoebox
# eval_loc : evaluation location(s) with shape ending in (3, ), representing x, y, z coords
# Other parameters:
# order : order of reflections to compute via ISM (must be at most the order of lattice_coord and lattice_ref)
# srate : sample rate
# rir_len : length of impulse response to output
# wall_filt_len : when mat_freqs is provided, the wall filter is computed via linear interpolation to DFT frequencies of length wall_filt_len
#                 if greater than 0.1 * rir_len is provided, the latter is used instead
# mach : speed of sound
# attenuation : acoustic attenuation constant in air, modeled as per https://ccrma.stanford.edu/~jos/pasp/Air_Absorption.html
# normalize : when set to True, chop off the part of the rir before the direct path
# normalize_mode : when set to 'batch', will preserve the relative direct path delays between output RIRs
#                  i.e. will chop off only the amount before the earliest direct path arrival;
#                  otherwise will normalize on a per-sample basis
# image_sigma : when set to a positive value, image source positions will be perturbed via a zero-mean uniform distribution with image_sigma range in all dimensions
# hpf : when true, use a 10 Hz highpass filter to remove DC bias from RIR
# render_tail : when set to True, render an approximate tail
def get_rir_lattice(lattice_coord, lattice_ref, eval_loc, order=REF_ORDER, srate=SRATE, rir_len=RIR_LEN, wall_filt_len=1024, mach=MACH, attenuation=ATTENUATION,
                    normalize=False, normalize_mode='each', image_sigma=0, hpf=True, render_tail=False):
    # construct RIR as a sum of delayed and scaled impulse responses
    wall_f_len = lattice_ref.shape[-1] if (lattice_ref.shape[-1] == 1) else min(int(0.1 * rir_len), wall_filt_len)
    max_delay = (rir_len - wall_f_len - 1) / srate # prevent circular shifting
    rfft_len = rir_len // 2 + 1
    impulse = torch.ones(rfft_len, requires_grad=False, device=lattice_coord.device)
    if (render_tail):
        timestamp = torch.arange(rir_len, requires_grad=False, device=lattice_coord.device) / srate
    omega = (-2.0j * torch.pi * srate / rir_len) * torch.arange(rfft_len, requires_grad=False, device=lattice_coord.device)
    direct_dist = torch.sqrt(torch.sum(torch.square(lattice_coord[...,0,:,0] - eval_loc), dim=-1, keepdim=True))
    if (normalize):
        if (normalize_mode == 'batch'):
            delay_offset = -1.0 * torch.min(direct_dist / mach)
            dist_norm = torch.min(direct_dist)
        else:
            delay_offset = -1.0 * direct_dist / mach
            dist_norm = direct_dist
    else:
        delay_offset = 0.0
        dist_norm = direct_dist
    # quadrant bits, for convenience in traversing lattice
    quadrant_b = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.intc)
    # compute image sources
    rir = 0.0
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
                        im_loc = torch.stack([lattice_coord[..., l_i, 0, quad_bi[0]],
                                              lattice_coord[..., l_j, 1, quad_bi[1]],
                                              lattice_coord[..., l_k, 2, quad_bi[2]]], dim=-1)
                        if (image_sigma > 0) and (l_i + l_j + l_k > 0):
                            im_loc = im_loc + image_sigma * (torch.rand_like(im_loc) - 0.5)
                        im_ref = (lattice_ref[..., l_i, 0, quad_bi[0],:] * lattice_ref[..., l_j, 1, quad_bi[1],:] * lattice_ref[..., l_k, 2, quad_bi[2],:])
                        delay, atten, dist = compute_path_t(im_loc, eval_loc, mach, attenuation)
                        if (render_tail and (l_i + l_j + l_k == order)):
                            max_order_amp.append((dist_norm / dist) * im_ref * atten)
                            max_order_delay.append(delay + delay_offset)
                        delay_mask = (delay + delay_offset >= 0) * (delay + delay_offset < max_delay)
                        rir = rir + delay_mask * (dist_norm / dist) * im_ref * atten * impulse * torch.exp(omega * (delay + delay_offset))
    rir = fft.irfft(rir, n=rir_len, norm='backward', dim=-1)
    # render approximate tail
    if (render_tail):
        # generate enveloped noise tails based on last image sources
        tail_func = lambda t, d, t0 : torch.exp(d * t) * (t > t0)
        tail_offset = 0.0 if normalize else delay_offset
        max_order_amp = torch.stack(max_order_amp, dim=-1)
        max_order_delay = torch.stack(max_order_delay, dim=-1)
        max_order_decay = safe_log(torch.abs(max_order_amp)) / max_order_delay
        tail_env = tail_func((timestamp + tail_offset)[...,None], max_order_decay, max_order_delay * (order + 1.0) / order)
        tail = torch.sum(0.5 * torch.randn(timestamp.shape + tail_env.shape[-1:], device=timestamp.device) * tail_env / max(6.0, order), dim=-1)
        rir = rir + tail
    if (hpf):
        b, a = iirfilter(2, 10.0, rp=5.0, rs=60.0, btype='highpass', ftype='butter', fs=SRATE, output='ba')
        a = torch.tensor(a, device=rir.device, dtype=rir.dtype)
        b = torch.tensor(b, device=rir.device, dtype=rir.dtype)
        rir = lfilter(rir, a, b)
    return rir

# Convenience function for computing the delay and attenuation corresponding to a given sound path
def compute_path_t(src_loc, mic_loc, mach, attenuation, keepdim=True):
    dist = torch.sqrt(torch.sum(torch.square(src_loc - mic_loc), dim=-1, keepdim=keepdim))
    delay = dist / mach
    atten = torch.exp(-1.0 * attenuation * dist)
    return delay, atten, dist

# Helper function for linear interpolation, taken from https://github.com/pytorch/pytorch/issues/50334#issuecomment-2304751532
def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int=-1, extrapolate: str='constant') -> torch.Tensor:
    """One-dimensional linear interpolation between monotonically increasing sample
    points, with extrapolation beyond sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: The :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: The :math:`x`-coordinates of the data points, must be increasing.
        fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
        dim: Dimension across which to interpolate.
        extrapolate: How to handle values outside the range of `xp`. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.

    Returns:
        The interpolated values, same size as `x`.
    """
    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1)
    xp = xp.movedim(dim, -1)
    fp = fp.movedim(dim, -1)

    m = torch.diff(fp) / torch.diff(xp) # slope
    b = fp[..., :-1] - m * xp[..., :-1] # offset
    indices = torch.searchsorted(xp, x, right=False)

    if extrapolate == 'constant':
        # Pad m and b to get constant values outside of xp range
        m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
        b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    else: # extrapolate == 'linear'
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

    values = m.gather(-1, indices) * x + b.gather(-1, indices)

    return values.movedim(-1, dim)

# Helper function to compute min-phase filter via Hilbert transform
def minphase(x_amp, out_size):
    assert (x_amp.shape[-1] == (out_size // 2 + 1)), "incompatible input and output sizes"
    ln_Xf = fft.irfft(safe_log(torch.abs(x_amp)), n=out_size, norm='backward')
    h = torch.zeros(out_size, device=x_amp.device)
    h[0] = 1
    h[1:x_amp.shape[-1]] = 2
    ln_x_hilbert = fft.rfft(ln_Xf * h, norm='backward')
    return x_amp * torch.exp(1.0j * torch.imag(ln_x_hilbert))

# Numerically stable natural log
def safe_log(x, eps=1e-9):
    return torch.log(torch.where(x <= eps, eps, x))
