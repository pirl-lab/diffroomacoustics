import numpy as np
import torch
from torch import fft
import torchaudio
from torchaudio import functional as taf

from defaults import *

# Loss function for comparing two RIRs
def rir_diff(x, y, win_len=512, win_stride=128):
    win = torch.blackman_window(window_length=512)
    x_tf = torch.stft(x, n_fft=win_len, hop_length=win_stride, window=win, return_complex=True)
    y_tf = torch.stft(y, n_fft=win_len, hop_length=win_stride, window=win, return_complex=True)
    x_mod = torch.abs(x_tf)
    y_mod = torch.abs(y_tf)
    x_arg = x_tf / torch.clamp(x_mod, min=EPS)
    y_arg = y_tf / torch.clamp(y_mod, min=EPS)
    x_mag = torch.log10(1.0 + x_mod)
    y_mag = torch.log10(1.0 + y_mod)
    return torch.mean(torch.square(x_mag - y_mag)) + torch.mean(torch.abs(x_arg - y_arg))

# Compute the definition (D) of an impulse response from get_rir with normalize=True
def definition(rir, srate=SRATE):
    rir_sq = torch.square(rir)
    idx_50ms = int(round(0.05 * srate))
    return torch.sum(rir_sq[:idx_50ms]) / torch.sum(rir_sq)

# Compute the clarity index (C80) of an impulse response from get_rir with normalize=True
def c80(rir, srate=SRATE):
    rir_sq = torch.square(rir)
    idx_80ms = int(round(0.08 * srate))
    return 10.0 * torch.log10(torch.sum(rir_sq[:idx_80ms]) / torch.sum(rir_sq[idx_80ms:]))

# Compute the center time of an impulse response from get_rir with normalize=True
def center_time(rir, srate=SRATE, rir_len=RIR_LEN):
    rir_sq = torch.square(rir)
    return torch.sum(rir_sq * torch.arange(rir_len) / srate) / torch.sum(rir_sq)

# Compute decay curve of an impulse respose from get_rir with normalize=True
def decay_curve(rir):
    rir_sq = torch.square(rir)
    crir = torch.cumsum(rir_sq, dim=0)
    decay = crir[-1] - crir
    return decay

# Compute reverb time of an impulse response from get_rir with normalize=True
# approximates T60 via decay curve as per Schroeder's method (integrated impulse response method)
def reverb_time(rir, tail_cutoff=0.25, srate=SRATE, eps=EPS):
    decay = decay_curve(rir[1:]) # remove direct path from computation
    idx_cutoff = int(round(tail_cutoff * srate))
    timestamp = torch.arange(idx_cutoff) / srate
    decay_norm = decay[:idx_cutoff] / decay[0]
    decay_log = torch.log10(torch.clamp(decay_norm, min=eps))
    ratio_60db = -6.0 # -60 dB
    # least squares regression to approximate exponential decay rate
    decay_rate = idx_cutoff * torch.sum(timestamp * decay_log) - torch.sum(timestamp) * torch.sum(decay_log)
    decay_rate = decay_rate / (idx_cutoff * torch.sum(torch.square(timestamp)) - torch.square(torch.sum(timestamp)))
    decay_time = ratio_60db / decay_rate
    return decay_time

# Compute modulation transfer function (MTF) from impulse response(s)
# returns the set of modulation transfer functions at all 14 modulation frequencies described by the STI standard
# rir : tensor of shape (L,) or (B,L) of B impulse responses of length L
# Returns: tensor of shape (14,) or (B,14) of the MTF evaluated at the 14 STI modulation frequencies
def mtf(rir, snr=None):
    if (len(rir.size()) < 2):
        rir = rir.unsqueeze(0)
        out_squeeze = True
    else:
        out_squeeze = False
    rir_sq = torch.square(rir).unsqueeze(1)
    noise_fact = 1.0
    if (snr is not None):
        noise_fact = noise_fact + 10.0**(-0.1*snr)
    timestamp = torch.arange(RIR_LEN) / SRATE
    timestamp = timestamp[None, None, :]
    f_mod = torch.tensor([0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5])
    f_mod = f_mod[None, :, None]
    transfer_f = torch.sum(torch.exp(-2.0j * torch.pi * f_mod * timestamp) * rir_sq, dim=2)
    transfer_f = (torch.abs(transfer_f) / torch.sum(rir_sq, dim=2)) / noise_fact
    if (out_squeeze):
        transfer_f = transfer_f.squeeze(0)
    return transfer_f

# 7-octave bandpass filterbank with the i-th central frequency at 125 * 2**i Hz
def octave_fbank(x, srate=SRATE, rir_len=RIR_LEN):
    x_fbank = torch.zeros((7, rir_len))
    for f_i in range(7):
        freq = 125.0 * (2**f_i)
        # compute Q-value for bandwidth = frequency
        omega = 2.0 * np.pi * freq / srate
        q = 1.0 / (2.0 * np.sinh(0.5 * np.log(2.0) * omega / np.sin(omega)))
        x_fbank[f_i,:] = taf.bandpass_biquad(x, srate, freq, q)
    return x_fbank

# Compute speech transmission index (STI) based on IEC 60268-16:2020
# rir : RIR in the same format output by get_rir
# snr : enviromental SNR level to evaluate STI with
# mtf_snr_range : dB value to clamp the absolute values of MTF SNR computations
def sti(rir, snr=SNR, mtf_snr_range=45):
    # TODO: perhaps use a soft clamping function?
    # makes sense if we're more concerned about maximizing / minimizing STI rather than achieving a target value,
    # though at that point may as well go all the way and find perhaps a reduction that has nicer derivatives
    mtf_snr_range = abs(mtf_snr_range)
    mtf_snr_ratio = mtf_snr_range / 10.0
    mtf_min = 10.0**(-mtf_snr_ratio) / (1.0 + 10.0**(-mtf_snr_ratio))
    mtf_max = 10.0**mtf_snr_ratio / (1.0 + 10**mtf_snr_ratio)
    rir_octave = octave_fbank(rir)
    mtf_octave = torch.clamp(mtf(rir_octave, snr=snr), min=mtf_min, max=mtf_max)
    mtf_snr = 10.0 * torch.log10(mtf_octave / (1.0 - mtf_octave))
    ti = (mtf_snr + mtf_snr_range) / (2.0 * mtf_snr_range)
    mti = torch.mean(ti, dim=1)
    sti_alpha = torch.tensor([0.085, 0.127, 0.23, 0.233, 0.309, 0.224, 0.173])
    sti_beta = torch.tensor([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])
    mti_cross = torch.sqrt(mti[:-1] * mti[1:])
    return torch.sum(sti_alpha * mti) + torch.sum(sti_beta * mti_cross)
