from gauge_eqn import GaussLawU1, build_hamiltonian, NetKetGaugeEq
from constants import LEARNING_RATE, SR_DIAG_SHIFT, ITER_W
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
from collections import defaultdict
from netket.hilbert import Fock
import netket as nk
import numpy as np
from time import time

def run_test(L, K, n_chains, g2, dtype, samples_per_chain, name: str):
    constraint = GaussLawU1(L, K)
    hilbert    = Fock(n_max=int(K-1), N=int(2*L*L), constraint=constraint)

    sampler = nk.sampler.MetropolisLocal(
        hilbert=hilbert, n_chains=n_chains, sweep_size=1, dtype=dtype
    )
    H = build_hamiltonian(hilbert, L=L, K=K, g2=g2)

    model = NetKetGaugeEq()
    vstate = nk.vqs.MCState(
        sampler=sampler, model=model, n_samples=samples_per_chain * n_chains,
    )
    opt = nk.optimizer.Adam(learning_rate=LEARNING_RATE)
    sr  = nk.optimizer.SR(diag_shift=SR_DIAG_SHIFT)
    vmc = nk.VMC(hamiltonian=H, optimizer=opt,
                    variational_state=vstate, preconditioner=sr)
    start = time()
    mem_usage = memory_usage((vmc.run, (), {'n_iter': ITER_W}))
    end = time()
    return (name, L, np.array(mem_usage), end - start) # TODO: Return the correct metric we are testing and not just L


def main():
    DEFAULT_L = 4  # default lattice size
    DEFAULT_K = 4  # default gauge group size
    DEFAULT_N_CHAINS = 8  # default number of chains
    SAMPLER_DTYPE = np.float32  # default sampler dtype
    SAMPLES_PER_CHAIN = 64
    lmin = 4  # minimum lattice size
    lmax = 8  # maximum lattice size

    test_L = np.linspace(lmin, lmax, lmax-lmin+1, dtype=np.int32)  # lattice sizes
    test_K = np.array([3, 4, 5, 6], dtype=np.int32)  # gauge group sizes
    test_chains = np.array([4, 8, 16, 32, 64], dtype=np.int32)  # number of chains
    g2 = 0.5  # coupling constant for testing
    results = []
    tests = []

    # Test Lattice Sizes
    tests += [(L, DEFAULT_K, DEFAULT_N_CHAINS, g2, SAMPLER_DTYPE, SAMPLES_PER_CHAIN, "L") for L in test_L]

    # Test Gauge Group Sizes
    tests += [(DEFAULT_L, K, DEFAULT_N_CHAINS, g2, SAMPLER_DTYPE, SAMPLES_PER_CHAIN, "K") for K in test_K]

    # Test Number of Chains
    tests += [(DEFAULT_L, DEFAULT_K, n_chains, g2, SAMPLER_DTYPE, SAMPLES_PER_CHAIN, "N_CHAINS") for n_chains in test_chains]

    for test in tests:
        try:
            result = run_test(*test)
            results.append(result)
            print(f"Test with L={test[0]}, K={test[1]}, n_chains={test[2]}: "
              f"Memory usage: {result[2].mean()} MB, Time taken: {result[3]:.2f} seconds")
        except Exception as e:
            print(f"Error running test with parameters {test}: {e}")
            continue

    to_plot = defaultdict(list)
    for result in results:
        to_plot[result[0]].append(result[1:])

    fig, axes = plt.subplots(2, len(to_plot), figsize=(15, 10))
    for (key, val), ax in zip(to_plot.items(), axes.T):
        for result in val:
            var, mem_usage, time_taken = result
            ax[0].plot(np.linspace(0, time_taken, mem_usage.size), mem_usage, marker='o', label=f'{key}={var}')
            ax[1].plot(var, time_taken, label=f'{key}={var}')
    plt.show()


if __name__ == "__main__":
    main()