# Gauge Equivariant Neural Networks for 2+1D U(1) Gauge Theory - Implementation Summary

## Paper Overview

**Title**: Gauge Equivariant Neural Networks for 2+1D U(1) Gauge Theory Simulations in Hamiltonian Formulation

**Main Contribution**: Development of gauge equivariant neural network wave function techniques for simulating continuous-variable quantum lattice gauge theories in the Hamiltonian formulation, specifically applied to 2+1D U(1) gauge theory for ground state finding using variational Monte Carlo.

## Key Technical Components for Reimplementation

### 1. Problem Setup - 2+1D U(1) Lattice Gauge Theory

**System Configuration**:
- L × L square lattice with periodic boundary conditions
- V = L² vertices, 2L² edges
- Gauge field: angular variables θ ∈ [0, 2π) on each edge
- Field configuration space: Ω = [0, 2π)^(2L²)
- Input representation: 2-channel image (x-axis and y-axis directions)

**Kogut-Susskind Hamiltonian**:
```
H = (g²/2e²) Σ_v (∂²/∂θ_{v,1}² + ∂²/∂θ_{v,2}²) - (2/g²) Σ_P cos(θ_{P,1} + θ_{P+e₁,2} - θ_{P+e₂,1} - θ_{P,2})
```
- First term: kinetic energy of gauge field
- Second term: magnetic energy
- g: coupling constant

### 2. Neural Network Architecture

**General Structure**:
- Trial wave function: ψ_ω = f_K ∘ ... ∘ f_1
- Gauge equivariant layers f_k (k ∈ {1, ..., K-1}): f_k(g·x) = g·f_k(x)
- Final gauge invariant layer f_K: operates on Wilson loops, f_K(g·x) = f_K(x)

**Two Network Variants**:
1. **Equ-NN**: 1 gauge equivariant block per equivariant layer
2. **Equ3-NN**: 3 gauge equivariant blocks per equivariant layer

**Architecture Parameters**:
- 2 equivariant layers
- 2 invariant layers  
- 2 equivariant features
- Invariant features: 4 (default), 3 for g² = 0.5, 0.6 on Equ-NN

**Activation Function - Complex ELU (cELU)**:
```python
def cELU(z):
    return ELU(Re(z)) + 1j * ELU(Im(z))

def ELU(x):
    return x if x > 0 else exp(x) - 1
```

### 3. Gauge Symmetry Implementation

**Gauge Transformation**: 
- On each vertex v: θ_{v,δ} → θ_{v,δ} + α_{v+e_δ} - α_v
- Wave function must be gauge invariant: g·ψ = ψ

**Wilson Loop Observable**:
```
W_C = ⟨ψ|cos(Σ_{v∈C} (θ_{v,1} + θ_{v+e₁,2} - θ_{v+e₂,1} - θ_{v,2}))|ψ⟩ / ⟨ψ|ψ⟩
```
Expected exponential decay: W_C = e^(-R₁×R₂ - 2a(R₁+R₂) + c)

### 4. Variational Monte Carlo Framework

**Energy Minimization**:
```
E = ⟨ψ_ω|H|ψ_ω⟩ / ⟨ψ_ω|ψ_ω⟩ = ∫ ψ_ω*(x) H ψ_ω(x) dx / ∫ ψ_ω*(x) ψ_ω(x) dx
```

**Gradient Computation**:
```
∂E/∂ω ≈ (2/N) Σ_{x~|ψ_ω|²} Re{E_loc(x) ∂/∂ω log ψ_ω(x)}
```
where E_loc(x) = Hψ_ω(x) / ψ_ω(x)

**Sampling and Optimization**:
- MCMC sampling: x ~ |ψ_ω|²
- Stochastic Reconfiguration method for optimization

### 5. Implementation Details

**System Sizes**:
- Strong coupling regime: L = 8, g ∈ [1, 4] with increment 0.1
- Weak coupling regime: L = 14, g ∈ [0.3, 0.9] with increment 0.1

**Data Flow**:
- Input: 2-channel image representing gauge field configuration
- Embedding: Ω = ∏_{e∈E} S¹ → ℂ^{N₁}
- Hidden spaces: ℂ^{N₂}, ..., ℂ^{N_K}, ℂ^{N_{K+1}} (where N_{K+1} = 1)
- Output: log(wave function amplitude)

### 6. Key Algorithms to Implement

#### A. Gauge Equivariant Layer Construction
- Ensure f_k(g·x) = g·f_k(x) property
- Handle complex-valued computations
- Implement proper gauge transformation handling

#### B. Wilson Loop Computation
- Calculate gauge-invariant observables
- Implement contour integration on lattice
- Handle periodic boundary conditions

#### C. MCMC Sampling
- Sample configurations according to |ψ_ω|²
- Implement Markov chain with proper acceptance ratios
- Handle complex-valued wave functions

#### D. Stochastic Reconfiguration
- Compute gradients of complex wave function
- Handle local energy E_loc(x) calculations
- Implement parameter updates with proper learning rates

#### E. Local Energy Calculation
- Apply Hamiltonian operator to wave function
- Handle kinetic energy terms (second derivatives)
- Compute magnetic energy terms (cosine interactions)

### 7. Performance Expectations

**Strong Coupling Regime (large g)**:
- Gauge equivariant networks outperform complex Gaussian wave functions
- Better performance as g approaches 1

**Weak Coupling Regime (small g)**:
- Comparable performance to complex Gaussian wave functions
- Better energy for g ≥ 0.6
- More challenging optimization in g → 0 limit

### 8. Implementation Challenges

1. **Complex-valued neural networks**: Proper gradient computation and parameter updates
2. **Gauge symmetry enforcement**: Ensuring equivariance/invariance properties
3. **MCMC efficiency**: Designing good proposal moves for gauge field configurations
4. **Local energy computation**: Efficient calculation of Hamiltonian action
5. **Numerical stability**: Handling exponential functions in ELU and energy calculations

### 9. Required Libraries/Components

- Complex-valued neural network framework
- MCMC sampling algorithms
- Automatic differentiation for complex functions
- Efficient tensor operations for lattice computations
- Visualization tools for Wilson loop analysis

## References

This summary is based on the paper "Gauge Equivariant Neural Networks for 2+1D U(1) Gauge Theory Simulations in Hamiltonian Formulation" (arXiv:2211.03198v1).