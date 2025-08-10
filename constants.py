import sys, site, jax, flax, netket
import numpy as np

# ============================== HYPERPARAMS ==================================
SEED = 0

# Lattice / discretisation
L = 4                 # lattice size (works for any L)
K = 4                 # U(1) discretisation (number of angle bins)

# Physics (electric-term finite-difference prefactor)
E_SCALE = 1.0         # set to (K/2π)^2 if you prefer continuum-ish spectrum

# Model / network
EQ_BLOCKS     = 1     # # equivariant blocks after the seed
INV_BLOCKS    = 1     # # invariant pooling blocks
FEATURES0     = 8     # channels after seed
FEATURES_GROW = 2     # widening factor per equivariant block
KERNEL        = (3, 3)
POOL          = (2, 2)

# Sampler / VMC (kept modest to avoid OOM)
N_CHAINS          = 8
SAMPLES_PER_CHAIN = 64
N_SAMPLES         = N_CHAINS * SAMPLES_PER_CHAIN
LEARNING_RATE     = 1e-3
SR_DIAG_SHIFT     = 1e-2
SAMPLER_DTYPE     = np.float32

# Optimisation steps
ITER_W       = 100    # for Wilson-loop optimisations
ITER_ENERGY  = 120    # for energy sweep

# Measurement batching (robust on a single GPU)
MEAS_WARMUP        = 200
MEAS_TOTAL_SAMPLES = 8000
MEAS_BATCH_SIZE    = 1024

# Couplings
G2_FOR_W = [0.4, 0.5, 0.6, 0.7]            # panels for Wilson loops
G2_FOR_E = np.linspace(1, 4, 10)        # ε(g²) sweep

# Rectangles (all minimal up to L/2 on an L×L periodic lattice)
RECT_MAX_SIDE = L // 2

# Derived
CHANNELS  = tuple(FEATURES0 * (FEATURES_GROW ** i) for i in range(EQ_BLOCKS + 1))
INV_DEPTH = INV_BLOCKS