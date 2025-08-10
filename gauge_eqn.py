# ============================== IMPORTS ======================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import jax
import jax.numpy as jnp
import flax.linen as nn

import netket as nk
from netket.hilbert import Fock
from netket.hilbert.constraint import DiscreteHilbertConstraint
from netket.operator import LocalOperator
from netket.utils import struct, dispatch
from constants import *

key = jax.random.PRNGKey(SEED)

# =========================== GAUSS-LAW CONSTRAINT ============================
@struct.dataclass
class GaussLawU1(DiscreteHilbertConstraint):
    L: int = struct.field(pytree_node=False)
    K: int = struct.field(pytree_node=False)

    def __init__(self, L, K):
        self.L=L
        self.K=K
        if L <= 0 or K <= 0:
            raise ValueError("L and K must be positive integers.")
        if K < 2:
            raise ValueError("K must be at least 2 for U(1) gauge theory.")
    
    def __hash__(self):
        return hash(("GaussLawU1", self.L, self.K))

    def __eq__(self, other):
        return isinstance(other, GaussLawU1) and (self.L, self.K) == (other.L, other.K)

    def __call__(self, x):
        ex = x[..., :self.L**2].reshape(*x.shape[:-1], self.L, self.L) - self.K//2
        ey = x[..., self.L**2:].reshape(*x.shape[:-1], self.L, self.L) - self.K//2
        div = (ex - jnp.roll(ex, 1, axis=-1) +
               ey - jnp.roll(ey, 1, axis=-2)) % self.K
        return jnp.all(div == 0, axis=(-2, -1))


# ============================== LATTICE HELPERS ==============================
def links_to_phases(state, L, K):
    """
    state: (…, 2L²) integers in 0..K-1
    returns o_x, o_y with shape (…, L, L)  (|o| = 1 complex phases)
    """
    phases = 2 * jnp.pi * state / K
    o_x = jnp.exp(1j * phases[...,        : L*L]).reshape(*state.shape[:-1], L, L)
    o_y = jnp.exp(1j * phases[..., L*L : 2*L*L]).reshape(*state.shape[:-1], L, L)
    return o_x, o_y


def wilson_plaquette(o_x, o_y):
    """
    cos(θ₁+θ₂−θ₃−θ₄) on each plaquette. Batch-safe.
    """
    θx = jnp.angle(o_x)
    θy = jnp.angle(o_y)
    θ1 = θx
    θ2 = jnp.roll(θy, shift=-1, axis=-1)   # +x
    θ3 = jnp.roll(θx, shift=-1, axis=-2)   # +y
    θ4 = θy
    return jnp.cos(θ1 + θ2 - θ3 - θ4)


def staple(o_x, o_y, direction="x"):
    """
    3-link open path (equivariant carrier). Batch-safe.
    """
    if direction == "x":                   # up → right → down†
        up      = jnp.roll(o_y, shift=-1, axis=-2)
        right   = jnp.roll(o_x, shift=-1, axis=-1)
        down_dg = jnp.conj(o_y)
        return up * jnp.conj(right) * down_dg
    else:                                  # right → up → left†
        right   = o_x
        up      = jnp.roll(o_y, shift=-1, axis=-2)
        left_dg = jnp.conj(jnp.roll(o_x, shift=-1, axis=-1))
        return right * up * left_dg
# ============================================================================


# ================================ HAMILTONIAN ================================
def plaquette_ops(hilbert, L, K, g2):
    """
    -(2/g²) * Σ_□ cos θ_□  (acts on 4 links per plaquette).
    """
    coupling = -(2.0 / g2)
    ops = []

    # Precompute diag(cos θ_sum) over K⁴
    val = np.zeros((K, K, K, K), dtype=np.float32)
    for n1 in range(K):
        for n2 in range(K):
            for n3 in range(K):
                for n4 in range(K):
                    angle = 2*np.pi*(n1+n2-n3-n4)/K
                    val[n1, n2, n3, n4] = np.cos(angle, dtype=np.float32)
    diag = np.diag(val.reshape(-1))

    for x in range(L):
        for y in range(L):
            idx1 =           y  * L +  x
            idx2 = L*L + ((y+1)%L)*L +  x
            idx3 =           y  * L + (x+1)%L
            idx4 = L*L +  y      *L +  x
            op = LocalOperator(hilbert, diag, [idx1, idx2, idx3, idx4])
            ops.append(coupling * op)
    return ops


def wilson_loop_ops(hilbert, L, K):
    """
    LocalOperators per plaquette with diagonal cos(θ_□) (no coupling).
    """
    val = np.zeros((K, K, K, K), dtype=np.float32)
    for n1 in range(K):
        for n2 in range(K):
            for n3 in range(K):
                for n4 in range(K):
                    angle = 2*np.pi*(n1+n2-n3-n4)/K
                    val[n1, n2, n3, n4] = np.cos(angle, dtype=np.float32)
    diag = np.diag(val.reshape(-1))

    ops = []
    for x in range(L):
        for y in range(L):
            idx1 =           y  * L +  x
            idx2 = L*L + ((y+1)%L)*L +  x
            idx3 =           y  * L + (x+1)%L
            idx4 = L*L +  y      *L +  x
            op = LocalOperator(hilbert, diag, [idx1, idx2, idx3, idx4])
            ops.append(op)
    return ops


def build_wilson_avg(hilbert, *, L, K):
    """(1/L²) Σ_plaquettes cos θ_□"""
    Wsum = sum(wilson_loop_ops(hilbert, L=L, K=K))
    return (1.0/(L*L)) * Wsum


def electric_ops(hilbert, *, L, K, g2):
    """
    Kinetic electric term in θ-basis:
      (g²/2) * E_SCALE * Δ_K
    where Δ_K is the discrete Laplacian on a K-point circle (non-diagonal).
    """
    D = np.zeros((K, K), dtype=np.float32)
    for n in range(K):
        D[n, n] = 2.0
        D[n, (n+1) % K] = -1.0
        D[n, (n-1) % K] = -1.0

    Ke = (g2/2.0) * E_SCALE * D   # non-diagonal K×K

    ops = []
    for link in range(2*L*L):
        ops.append(LocalOperator(hilbert, Ke, [link]))
    return ops


def build_hamiltonian(hilbert, L, K, g2):
    H_e = sum(electric_ops(hilbert, L=L, K=K, g2=g2))
    H_b = sum(plaquette_ops(hilbert, L=L, K=K, g2=g2))
    return H_e + H_b


# ============================== MODEL / NETWORK ==============================
def c_elu(z):
    """Complex ELU activation (papers cELU)."""
    return jax.nn.elu(jnp.real(z)) + 1j * jax.nn.elu(jnp.imag(z))

def complex_conv(x_r, x_i, features, kernel, name):
    conv_r = nn.Conv(features, kernel, name=f"{name}_r")
    conv_i = nn.Conv(features, kernel, name=f"{name}_i")
    real = conv_r(x_r) - conv_i(x_i)
    imag = conv_r(x_i) + conv_i(x_r)
    return real + 1j*imag

class GaugeEqNet(nn.Module):
    channels: tuple = CHANNELS
    inv_depth: int = INV_DEPTH

    @nn.compact
    def __call__(self, o_x, o_y):
        # 1) invariant seed on plaquettes
        W = wilson_plaquette(o_x, o_y)  # (B,L,L), real
        x = complex_conv(W[..., None], jnp.zeros_like(W)[..., None],
                         self.channels[0], KERNEL, name="seed")

        # 2) equivariant stack
        for k, Cout in enumerate(self.channels[1:]):
            h = complex_conv(jnp.real(W)[..., None], jnp.imag(W)[..., None],
                             Cout, KERNEL, name=f"equiv_conv_{k}")
            h = c_elu(h)
            P_x = staple(o_x, o_y, "x")[..., None]
            P_y = staple(o_x, o_y, "y")[..., None]
            x = (P_x * h + P_y * h) / 2.0

        # 3) invariant pooling blocks
        for j in range(self.inv_depth):
            inv = complex_conv(jnp.real(W)[..., None], jnp.imag(W)[..., None],
                               x.shape[-1], KERNEL, name=f"inv_conv_{j}")
            inv = c_elu(inv)
            inv = jax.lax.reduce_window(inv, 0.0, jax.lax.add,
                                        (1, POOL[0], POOL[1], 1),
                                        (1, 1, 1, 1), "SAME") / float(POOL[0]*POOL[1])
            x = inv

        # 4) global sum & complex readout: RETURN logψ (not ψ!)
        x = jnp.sum(x, axis=(-3, -2))
        feats = jnp.concatenate([jnp.real(x), jnp.imag(x)], axis=-1)
        log_amp = nn.Dense(1, use_bias=False, name="readout_amp")(feats)[..., 0]
        phi_raw = nn.Dense(1, use_bias=False, name="readout_phase")(feats)[..., 0]
        phase   = jnp.tanh(phi_raw) * jnp.pi    # bounded phase
        logpsi  = log_amp + 1j * phase          # <-- NetKet expects logψ
        return logpsi

class NetKetGaugeEq(nn.Module):
    @nn.compact
    def __call__(self, flat_int_cfg):   # (B, 2L²)
        B = flat_int_cfg.shape[0]
        o_x, o_y = links_to_phases(flat_int_cfg, L=L, K=K)
        o_x = o_x.reshape((B, L, L))
        o_y = o_y.reshape((B, L, L))
        net = GaugeEqNet()
        return net(o_x, o_y)


# ===================== RANDOM GAUGE-INVARIANT STATES ========================
@dispatch.dispatch
def random_state(hilb: Fock,
                 constraint: GaussLawU1,
                 key,
                 batches: int,
                 *,
                 dtype=None):
    """
    Draw `batches` independent gauge-invariant U(1) configs in one shot.
    """
    L, K = constraint.L, constraint.K
    key_alpha, _ = jax.random.split(key)

    alpha = jax.random.randint(key_alpha, (batches, L, L), 0, K)
    n_x = (alpha - jnp.roll(alpha, -1, axis=-1)) % K
    n_y = (alpha - jnp.roll(alpha, -1, axis=-2)) % K

    states = jnp.concatenate([n_x.reshape(batches, -1),
                              n_y.reshape(batches, -1)], axis=1)

    return states.astype(dtype or jnp.int32)


# ========================= WILSON LOOP MEASUREMENT ==========================
def rect_wilson_avg_batch(o_x, o_y, r1, r2):
    """
    <Re ∏_C U> = <cos(Σθ)> using link PRODUCTS (gauge-correct).
    o_x, o_y: (B, L, L); returns (B,)
    """
    top = 1.0 + 0.0j
    for k in range(r1):
        top *= jnp.roll(o_x, -k, axis=-1)
    right = 1.0 + 0.0j
    for k in range(r2):
        right *= jnp.roll(jnp.roll(o_y, -r1, axis=-1), -k, axis=-2)
    left = 1.0 + 0.0j
    for k in range(r1):
        left *= jnp.conj(jnp.roll(jnp.roll(o_x, -r2, axis=-2), -k, axis=-1))
    down = 1.0 + 0.0j
    for k in range(r2):
        down *= jnp.conj(jnp.roll(o_y, -k, axis=-2))

    U_rect = top * right * left * down
    return jnp.mean(jnp.real(U_rect), axis=(-2, -1))


def measure_rectangles(vstate: nk.vqs.MCState, rectangles, L, K,
                       warmup=MEAS_WARMUP,
                       total_samples=MEAS_TOTAL_SAMPLES,
                       batch_size=MEAS_BATCH_SIZE):
    """
    Robust, batched estimator of <W(r1,r2)> to avoid OOM.
    """
    # warmup/ decorrelate
    vstate.sample(chain_length=warmup)

    total_samples = int(max(total_samples, 8*N_CHAINS))
    batch_size    = int(max(batch_size,    4*N_CHAINS))

    all_vals = {rc: [] for rc in rectangles}
    remain = total_samples
    while remain > 0:
        this = min(batch_size, remain)
        vstate.sample(n_samples=this)

        S  = np.asarray(vstate.samples).reshape(-1, 2*L*L)
        o_x, o_y = links_to_phases(S, L=L, K=K)
        for (r1, r2) in rectangles:
            v = np.asarray(rect_wilson_avg_batch(o_x, o_y, r1, r2))
            all_vals[(r1, r2)].append(v)
        remain -= this

    out = {}
    for rc, chunks in all_vals.items():
        v = np.concatenate(chunks, axis=0)
        mean = float(v.mean())
        err  = float(v.std(ddof=1)/np.sqrt(v.size))
        out[rc] = (mean, err)
    return out

# ================================== MAIN ====================================
if __name__ == "__main__":
    constraint = GaussLawU1(L, K)
    hilbert    = Fock(n_max=K-1, N=2*L*L, constraint=constraint)
    print("Hilbert size:", hilbert.size)
    # quick sanity: centered config satisfies Gauss law
    state0 = jnp.full((2*L*L,), K//2, dtype=jnp.int32)
    assert constraint(state0), "Gauss law must hold on centered state."

    # rectangles (all minimal up to L/2)
    rectangles = [(r1, r2) for r1 in range(1, RECT_MAX_SIDE+1)
                           for r2 in range(1, RECT_MAX_SIDE+1)]
    areas = sorted({r1*r2 for (r1, r2) in rectangles})

    # ---------------- Wilson loops for selected g² ----------------
    curves = {}
    sampler = nk.sampler.MetropolisLocal(
        hilbert=hilbert, n_chains=N_CHAINS, sweep_size=1, dtype=SAMPLER_DTYPE
    )
    for g2 in G2_FOR_W:
        print(f"\n=== Wilson loops at g^2 = {g2} ===")
        H = build_hamiltonian(hilbert, L=L, K=K, g2=g2)

        vstate = nk.vqs.MCState(
            sampler=sampler, model=NetKetGaugeEq(), n_samples=N_SAMPLES
        )
        opt = nk.optimizer.Adam(learning_rate=LEARNING_RATE)
        sr  = nk.optimizer.SR(diag_shift=SR_DIAG_SHIFT)
        vmc = nk.VMC(hamiltonian=H, optimizer=opt,
                     variational_state=vstate, preconditioner=sr)
        vmc.run(n_iter=ITER_W, out=f"outputs/vmc_W_g2_{g2}")

        # measure and aggregate by area
        m = measure_rectangles(vstate, rectangles, L=L, K=K)
        mean_by_area, err_by_area = [], []
        for A in areas:
            vals = [m[(r1,r2)][0] for (r1,r2) in rectangles if r1*r2 == A]
            errs = [m[(r1,r2)][1] for (r1,r2) in rectangles if r1*r2 == A]
            mean_by_area.append(np.mean(vals))
            err_by_area.append(np.sqrt(np.sum(np.array(errs)**2)) / max(1, len(errs)))
        curves[g2] = (np.array(mean_by_area), np.array(err_by_area))

    # shared y-limits (decade aligned) for all WL panels
    all_y = np.concatenate([curves[g2][0] for g2 in G2_FOR_W])
    all_e = np.concatenate([curves[g2][1] for g2 in G2_FOR_W])
    y_floor = max(1e-4, float(np.min(np.clip(all_y - all_e, 1e-12, None))))
    y_ceil  = float(np.max(all_y + all_e))
    y_lo = 10.0 ** np.floor(np.log10(y_floor))
    y_hi = 10.0 ** np.ceil(np.log10(y_ceil))

    for g2 in G2_FOR_W:
        y, e = curves[g2]
        mask_nonpos = (y <= 0)
        y_plot = np.where(mask_nonpos, y_floor, y)

        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        ax.set_yscale('log')
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(min(areas)-0.5, max(areas)+0.5)
        ax.set_xticks(areas)

        ax.errorbar(areas, y_plot, yerr=e, fmt='o', ms=4, lw=0,
                    capsize=4, elinewidth=1)

        if np.any(mask_nonpos):
            ax.scatter(np.array(areas)[mask_nonpos], y_plot[mask_nonpos],
                       facecolors='none', edgecolors='r', s=40, linewidths=1.0,
                       label='≤ 0 (clipped)')

        decades = [d for d in [1e-4,1e-3,1e-2,1e-1,1e0] if y_lo <= d <= y_hi]
        if decades: ax.set_yticks(decades)
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(which='major', alpha=0.3)

        # dashed exponential fit on positive points only
        ok = (y > 0)
        if np.count_nonzero(ok) >= 2:
            w = 1.0/np.maximum(e[ok], 1e-6)
            c1, c0 = np.polyfit(np.array(areas)[ok], np.log(y[ok]), 1, w=w)
            ax.plot(areas, np.exp(c0 + c1*np.array(areas)), '--', alpha=0.6)

        ax.set_xlabel(r'$R_1 \times R_2$ (area)')
        ax.set_ylabel(r'$\langle W(R_1,R_2)\rangle$')
        ax.set_title(rf'Wilson loops vs area  (L={L}, $g^2={g2}$)')
        if np.any(mask_nonpos): ax.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    # ---------------- energy density ε(g²) sweep -------------------
    eps_mean, eps_err = [], []
    for g2 in G2_FOR_E:
        H = build_hamiltonian(hilbert, L=L, K=K, g2=g2)

        vstate = nk.vqs.MCState(
            sampler=sampler, model=NetKetGaugeEq(), n_samples=N_SAMPLES,
        )
        opt = nk.optimizer.Adam(learning_rate=LEARNING_RATE)
        sr  = nk.optimizer.SR(diag_shift=SR_DIAG_SHIFT)
        vmc = nk.VMC(hamiltonian=H, optimizer=opt,
                     variational_state=vstate, preconditioner=sr)
        vmc.run(n_iter=ITER_ENERGY)

        stats = vstate.expect(H)  # complex logψ -> real energy; imag ~ 0
        eps_mean.append(float(np.asarray(stats.mean).real) / (L*L))
        eps_err .append(float(np.asarray(stats.error_of_mean).real) / (L*L))

    g2_arr = np.asarray(G2_FOR_E)
    eps    = np.asarray(eps_mean)
    err    = np.asarray(eps_err)

    plt.figure(figsize=(5.2, 3.4))
    plt.errorbar(g2_arr, eps, yerr=err, fmt='o-', capsize=3)
    plt.xlabel(r'$g^{2}$')
    plt.ylabel(r'$\varepsilon \;=\; E/L^{2}$')
    plt.title(rf'Variational energy density vs coupling  (L={L})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()