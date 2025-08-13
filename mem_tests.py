from gauge_eqn import GaussLawU1, build_hamiltonian, NetKetGaugeEq
from constants import LEARNING_RATE, SR_DIAG_SHIFT, ITER_W
import matplotlib.pyplot as plt
import numpy as np
from time import time
from utils import GPUMemProf, SystemMemProf
from collections import defaultdict
from netket.hilbert import Fock
import netket as nk
import numpy as np
from time import time
from utils import GPUMemProf
import psutil


def get_per_gpu(gpu_mem):
    time = []
    gpus = defaultdict(list)
    gpu_max_mem = []

    for sample in gpu_mem:
        time.append(sample[0])
        if not len(gpu_max_mem):
            gpu_max_mem += [x[1] for x in sample[1]]
        for i, gpu in enumerate(sample[1]):
            gpus[i].append(gpu[0])
    return np.array(time), gpus, np.array(gpu_max_mem)


def plot_gpu_mem(gpu_mem, ax: plt.Axes):
    time, gpus, gpu_max_mem = get_per_gpu(gpu_mem)
    time = np.array(time)
    time = time - time.min()
    for gpu, gpu_mem_trace in gpus.items():
        ax.plot(time, gpu_mem_trace, label=f"GPU {gpu}")
    ax.hlines(y=gpu_max_mem[0], xmin=time.min(), xmax=time.max(), label="Max Memory", linestyles='-', colors='orange')
    ax.legend()
    plt.savefig("gpu_mem.png")


def plot_sys_mem(mem_samples, ax: plt.Axes):
    time = np.array([x[0] for x in mem_samples])
    time = time - time.min()
    mem = [x[1] for x in mem_samples]
    ax.plot(time, mem, label="System Memory")
    max_mem = psutil.virtual_memory().total / (1024**3)
    ax.hlines(y=max_mem, xmin=time.min(), xmax=time.max(), label="Max Memory", linestyles='-', colors='orange')
    ax.legend()
    plt.savefig("sys_mem.png")


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
    gpu_prof = GPUMemProf(2)
    sys_prof = SystemMemProf(0.2)
    gpu_prof.start()
    sys_prof.start()
    start = time()
    vmc.run(n_iter=ITER_W)
    end = time()
    gpu_mem = gpu_prof.stop()
    sys_mem = sys_prof.stop()
    plot_gpu_mem(gpu_mem, plt.gca())
    plot_sys_mem(sys_mem, plt.gca())
    # keep the same shape as previous return: convert sys_mem values to array of MB
    mem_usage = np.array([v for (_, v) in sys_mem], dtype=float)
    return (name, L, mem_usage, end - start) # TODO: Return the correct metric we are testing and not just L


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
        # try:
            result = run_test(*test)
            results.append(result)
            print(f"Test with L={test[0]}, K={test[1]}, n_chains={test[2]}: "
              f"Memory usage: {result[2].mean()} MB, Time taken: {result[3]:.2f} seconds")
            return
        # except Exception as e:
        #     print(f"Error running test with parameters {test}: {e}")
        #     continue

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