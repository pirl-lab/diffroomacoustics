import torch

SRATE = 48000 # sample rate
MACH = 343.0 # speed of sound
ATTENUATION = 0.0015 # approx. acoustic attenuation constant in air, modeled as per https://ccrma.stanford.edu/~jos/pasp/Air_Absorption.html
RIR_LEN = SRATE * 2 # (fixed) max length of rir
REF_ORDER = 15 # order of reflections to compute

IMPULSE_SIGMA = 0.0075 # gaussian width in seconds

EPS = torch.finfo(torch.float32).eps # machine epsilon
SNR = 6.0 # SNR level to evaluate STI with
