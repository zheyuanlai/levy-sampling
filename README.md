# LSB-MC: Lévy-Score-Based Monte Carlo for Boltzmann Sampling

## Overview

Sampling from Boltzmann distributions of the form
$$
p_\infty(x) = Z^{-1} \exp\left\{-\frac{2V(x)}{\sigma^2}\right\}
$$
via overdamped Langevin dynamics becomes prohibitively slow when the potential $V$ features multiple deep wells separated by high energy barriers. LSB-MC addresses this by augmenting the standard diffusion process with **compound Poisson jumps** and a **stationary Lévy-score correction**, enabling macroscopic spatial transitions while preserving the target invariant measure.

## Sampling Methods

This repository implements four sampling algorithms targeting the same Boltzmann distribution. All methods are compared on identical test problems to assess convergence rates and exploration efficiency.

---

### 1. ULA (Unadjusted Langevin Algorithm)

**Target Distribution**:
$$
p_\infty(x) = Z^{-1} \exp\left\{-\frac{2V(x)}{\sigma^2}\right\}, \quad x \in \mathbb{R}^d
$$

**Continuous-Time SDE**:
$$
dX_t = -\nabla V(X_t) \, dt + \sigma \, dB_t
$$
where $B_t$ is standard Brownian motion and $\sigma > 0$ is the noise intensity.

**Discrete-Time Update** (Euler-Maruyama):
$$
X_{n+1} = X_n + \frac{dt \cdot (-\nabla V(X_n))}{1 + dt \|\nabla V(X_n)\|} + \sigma \sqrt{dt} \, Z_n, \quad Z_n \sim \mathcal{N}(0, I)
$$
The denominator $1 + dt \|\nabla V(X_n)\|$ implements **taming** for numerical stability when gradients are large.

---

### 2. MALA (Metropolis-Adjusted Langevin Algorithm)

**Target Distribution**: 
$$
p_\infty(x) = Z^{-1} \exp\left\{-\frac{2V(x)}{\sigma^2}\right\}, \quad x \in \mathbb{R}^d
$$

**Proposal Distribution**:
$$
Y = X + \frac{1}{2} dt \, \nabla \log p_\infty(X) + \sqrt{dt} \, Z, \quad Z \sim \mathcal{N}(0, I)
$$
where $\nabla \log p_\infty(x) = -2\nabla V(x)/\sigma^2$.

**Acceptance Probability**:
$$
\alpha(X, Y) = \min\left\{1, \frac{p_\infty(Y) \, q(X \mid Y)}{p_\infty(X) \, q(Y \mid X)}\right\}
$$
where $q(Y \mid X)$ is the Gaussian proposal kernel:
$$
q(Y \mid X) \propto \exp\left\{-\frac{\|Y - X - \tfrac{1}{2}dt \, \nabla \log p_\infty(X)\|^2}{2dt}\right\}
$$

**Discrete-Time Update**:
1. Propose $Y$ from the above Gaussian
2. Compute $\log \alpha = (\log p_\infty(Y) + \log q(X \mid Y)) - (\log p_\infty(X) + \log q(Y \mid X))$
3. Accept $Y$ with probability $\min\{1, e^{\log \alpha}\}$; otherwise retain $X$


---

### 3. FLMC (Fractional Langevin Monte Carlo)

Adapted from [Fractional Langevin Monte Carlo: Exploring Levy Driven Stochastic Differential Equations for Markov Chain Monte Carlo](https://arxiv.org/abs/1706.03649) (ICML 2017).

**Target Distribution**: 
$$
p_\infty(x) = Z^{-1} \exp\left\{-\frac{2V(x)}{\sigma^2}\right\}, \quad x \in \mathbb{R}^d
$$

**Continuous-Time SDE**:
$$
dX_t = -c_\alpha \nabla V(X_t) \, dt + \sigma \, dt^{1/\alpha} \, dL_t^\alpha
$$
where:
- $L_t^\alpha$ is a symmetric $\alpha$-stable Lévy process with tail index $\alpha \in (1, 2]$
- $c_\alpha = \Gamma(\alpha - 1) / \Gamma(\alpha/2)^2$ is a normalization constant

**Discrete-Time Update**:
$$
X_{n+1} = X_n + \frac{dt \cdot (-c_\alpha \nabla V(X_n))}{1 + dt \|c_\alpha \nabla V(X_n)\|} + \sigma \, dt^{1/\alpha} \, Z_n
$$
where $Z_n$ is sampled from a symmetric $\alpha$-stable distribution using the **Chambers-Mallows-Stuck algorithm**.

**Alpha-Stable Noise**:
- **Low-dimensional (1D/2D)**: Coordinatewise independent $\alpha$-stable samples
- **High-dimensional ($d \geq 3$)**: Genuinely isotropic $\alpha$-stable vectors $Z = R \cdot U$, where:
  - $U \sim \text{Uniform}(\mathbb{S}^{d-1})$ (random direction on unit sphere)
  - $R \sim S_\alpha^{1/\alpha}$ (radial component with $\alpha$-stable distribution)

---

### 4. LSB-MC (Lévy-Score-Based Monte Carlo)

**Target Distribution**: 
$$
p_\infty(x) = Z^{-1} \exp\left\{-\frac{2V(x)}{\sigma^2}\right\}, \quad x \in \mathbb{R}^d
$$

**Continuous-Time SDE**:
$$
dZ_t = \left(-\nabla V(Z_{t-}) + S_L^s(Z_{t-})\right) dt + \sigma \, dB_t + dL_t
$$
where:
- $S_L^s(x)$ is the **stationary Lévy-score correction**:
$$
S_L^s(x) = -\int_0^1 \int_{\mathbb{R}^d \setminus \{0\}} r \exp\left\{-\frac{2(V(x - \theta r) - V(x))}{\sigma^2}\right\} \nu(dr) \, d\theta
$$
- $\nu$ is the Lévy measure governing the jump law
- $L_t$ is a pure-jump Lévy process with measure $\nu$

**Compound Poisson Jump Law**:
In this implementation, $\nu = \lambda \, \nu_J$ where:
- $\lambda > 0$ is the **jump intensity** (expected number of jumps per unit time)
- $\nu_J$ is a **discrete mixture of isotropic jumps**:
$$
\nu_J = \sum_{k=1}^K p_k \, \delta_{m_k \sigma_L}
$$
with:
  - $\{m_1, \ldots, m_K\}$: jump **multipliers** (e.g., $[1.0, 1.8, 2.6]$)
  - $\{p_1, \ldots, p_K\}$: probability masses ($\sum_k p_k = 1$)
  - $\sigma_L > 0$: base jump **magnitude scale**

**Jump Direction Sampling**:
- **Low-dimensional (1D/2D)**: Jumps are sampled as $\pm m_k \sigma_L$ (random sign)
- **High-dimensional ($d \geq 3$)**: Jumps are **genuinely isotropic** in $\mathbb{R}^d$:
$$
\text{Jump vector} = (m_k \sigma_L) \cdot U, \quad U \sim \text{Uniform}(\mathbb{S}^{d-1})
$$
  This ensures rotational invariance (not coordinatewise composition).

**Discrete-Time Update**:
1. Compute drift: $b(x) = -\nabla V(x) + S_L^s(x)$
2. Tamed diffusion: $x_{\text{new}} = x + \frac{dt \cdot b(x)}{1 + dt \|b(x)\|} + \sigma \sqrt{dt} Z_{\text{diff}}$
3. Sample jumps: $N_{\text{jumps}} \sim \text{Poisson}(\lambda dt)$
4. For each jump: sample multiplier $m_k$ with probability $p_k$, sample direction $U$, add jump $m_k \sigma_L \cdot U$

**Stationary Lévy-Score Precomputation**:
The integral defining $S_L^s(x)$ is approximated via:
- **Trapezoidal quadrature** in $\theta \in [0, 1]$ (typically $n_\theta = 7$ to $23$ points)
- **Monte Carlo or antithetic sampling** over directions (typically $n_{\text{dir}} = 8$ directions for high-dim)

For 1D/2D, the score is precomputed on a spatial grid and interpolated during simulation.

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| $V(x)$ | Potential function |
| $\nabla V(x)$ | Gradient of potential |
| $p_\infty(x)$ | Target Boltzmann distribution |
| $\sigma$ or $\epsilon$ | Noise scale (controls temperature) |
| $dt$ | Time step size |
| $\alpha$ | FLMC tail index ($\alpha \in (1, 2]$) |
| $c_\alpha$ | FLMC drift normalization constant |
| $\lambda$ | LSB-MC jump intensity |
| $\sigma_L$ | LSB-MC base jump magnitude |
| $\{m_k\}$ | LSB-MC jump multipliers |
| $\{p_k\}$ | LSB-MC multiplier probabilities |
| $S_L^s(x)$ | Stationary Lévy-score correction |
| $\nu(dr)$ | Lévy measure |

---

## Numerical Experiments

All experiments compare the four methods (ULA, MALA, FLMC, LSB-MC) on benchmark problems with varying complexity. Metrics track convergence to the target distribution over time.

### 1. Ginzburg–Landau Double-Well Potential (1D)

**Mathematical Definition**:
$$
V(x) = \frac{1}{4} x^4 - \frac{1}{2} x^2
$$

**Gradient**:
$$
\nabla V(x) = x^3 - x = x(x^2 - 1)
$$

**Target Distribution**:
$$
p_\infty(x) \propto \exp\left\{-\frac{2V(x)}{\sigma^2}\right\}
$$

**SDE**:
$$
dX_t = (X_t - X_t^3) \, dt + \epsilon \, dB_t
$$
where $\epsilon = \sigma$ under the global convention.

**Metrics**:
- **Wasserstein-2 distance** ($W_2$): Exact 1D computation via quantile matching
- **$L^1$ error**: $\int |p_{\text{emp}}(x) - p_\infty(x)| dx$
- **$L^2$ error**: $\sqrt{\int (p_{\text{emp}}(x) - p_\infty(x))^2 dx}$
- **Average Bias**: $|\mathbb{E}[X_{\text{emp}}] - \mathbb{E}[X_{\text{true}}]|$ for observable $\mathbb{E}[X]$

**Hyperparameters** (`doublewell.py`):
```python
sigma = 0.35          # Noise scale (weakly stochastic regime)
dt = 0.005            # Time step
T = 40.0              # Total simulation time
N = 8000              # Number of particles
alpha = 1.5           # FLMC tail index
lam = 1.0             # LSB-MC jump intensity
sigma_L = 1.6         # LSB-MC base jump magnitude
multipliers = [1.0, 1.8, 2.6]   # LSB-MC jump size multipliers
pm = [0.70, 0.22, 0.08]         # LSB-MC multiplier probabilities
```

**LSB-MC Jump Details**:
- **1D jump law**: Jumps are $\pm m_k \sigma_L$ with random sign
- **Score precomputation**: $n_\theta = 23$ quadrature points in $\theta \in [0, 1]$
- Precomputed on spatial grid $[-3, 3]$ with $600$ points, then interpolated

---

### 2. Ring Potential (2D)

**Mathematical Definition**:
$$
V(x, y) = \left(1 - x^2 - y^2\right)^2 + \frac{y^2}{x^2 + y^2}
$$

**Target Distribution**:
$$
p_\infty(x, y) \propto \exp\left\{-\frac{2V(x, y)}{\sigma^2}\right\}
$$

**SDE**:
$$
\begin{cases}
dX_t = -\partial_x V(X_t, Y_t) \, dt + \sqrt{\epsilon} \, dB_t^{(1)} \\
dY_t = -\partial_y V(X_t, Y_t) \, dt + \sqrt{\epsilon} \, dB_t^{(2)}
\end{cases}
$$
where $\sigma = \sqrt{\epsilon}$ (matches global convention).

**Metrics**:
- **$L^1$ error**: Mean absolute error on $200 \times 200$ grid
- **$L^2$ error**: Root mean squared error (RMSE) on grid

**Hyperparameters** (`ring.py`):
```python
eps = 0.35           # Noise parameter (σ² under global convention)
dt = 0.0015          # Time step
T = 40.0             # Total time
N = 15000            # Number of particles
alpha = 1.5          # FLMC tail index
lam = 1.6            # LSB-MC jump intensity
sigma_L = 1.25       # LSB-MC base jump magnitude
multipliers = [1.0, 1.7, 2.4]     # LSB-MC jump multipliers
pm = [0.70, 0.22, 0.08]           # LSB-MC multiplier probabilities
```

**LSB-MC Jump Details**:
- **2D jump law**: Jumps are $m_k \sigma_L \cdot (\cos\theta, \sin\theta)$ with $\theta \sim \text{Uniform}[0, 2\pi]$ (isotropic in plane)
- **Score precomputation**: $200 \times 200$ spatial grid, $n_\theta = 7$, antithetic direction sampling

---

### 3. Four-Well Potential (2D)

**Mathematical Definition**:
$$
V(x, y) = (x^2 - 1)^2 + (y^2 - 1)^2
$$

**Target Distribution**:
$$
p_\infty(x, y) \propto \exp\left\{-\frac{V(x, y)}{\epsilon^2}\right\}
$$

**SDE**:
$$
\begin{cases}
dX_t = -\tfrac{1}{2} \partial_x V(X_t, Y_t) \, dt + \epsilon \, dB_t^{(1)} \\
dY_t = -\tfrac{1}{2} \partial_y V(X_t, Y_t) \, dt + \epsilon \, dB_t^{(2)}
\end{cases}
$$
Note the factor $1/2$ in the drift, corresponding to an effective potential $V_{\text{eff}} = V/2$.

**Hyperparameters** (`fourwells.py`):
```python
eps = 0.35           # Noise scale
dt = 0.005           # Time step
T = 15.0             # Total time
N = 16000            # Number of particles
alpha = 1.5          # FLMC tail index
lam = 1.4            # LSB-MC jump intensity
sigma_L = 1.3        # LSB-MC base jump magnitude
multipliers = [1.0, 1.7, 2.4]
pm = [0.70, 0.22, 0.08]
```

---

### 4. Müller–Brown Potential (2D)

**Mathematical Definition**:
$$
V(x, y) = \sum_{i=1}^4 A_i \cdot s \cdot \exp\left( a_i (x - x_i)^2 + b_i (x - x_i)(y - y_i) + c_i (y - y_i)^2 \right)
$$
with scale factor $s = 0.05$ and parameters:

| $i$ | $A_i$ | $a_i$ | $b_i$ | $c_i$ | $x_i$ | $y_i$ |
|-----|-------|-------|-------|-------|-------|-------|
| 1   | $-200$ | $-1$ | $0$ | $-10$ | $1$ | $0$ |
| 2   | $-200$ | $-1$ | $0$ | $-10$ | $0$ | $0.5$ |
| 3   | $-200$ | $-6.5$ | $11$ | $-6.5$ | $-0.5$ | $1.5$ |
| 4   | $-200$ | $-3$ | $0$ | $-3$ | $-0.8$ | $-0.5$ |

**Target Distribution**:
$$
p_\infty(x, y) \propto \exp\left\{-\frac{V(x, y)}{\epsilon^2}\right\}
$$

**SDE**:
$$
\begin{cases}
dX_t = -\tfrac{1}{2} \partial_x V(X_t, Y_t) \, dt + \epsilon \, dB_t^{(1)} \\
dY_t = -\tfrac{1}{2} \partial_y V(X_t, Y_t) \, dt + \epsilon \, dB_t^{(2)}
\end{cases}
$$

**Hyperparameters** (`mueller.py`):
```python
eps = 0.0015         # Noise scale (very small, highly metastable)
dt = 0.0002          # Time step
T = 2.0              # Total time
N = 20000            # Number of particles
alpha = 1.5          # FLMC tail index
lam = 1.5            # LSB-MC jump intensity
sigma_L = 0.25       # LSB-MC base jump magnitude (smaller due to small eps)
multipliers = [1.0, 1.8, 2.6]
pm = [0.70, 0.22, 0.08]
```

---

### 5. Lennard–Jones Cluster (High-Dimensional)

**Mathematical Definition**:
For $N$ particles in $\mathbb{R}^3$, positions $R = (r_1, \ldots, r_N) \in (\mathbb{R}^3)^N$:
$$
V(R) = \sum_{1 \leq i < j \leq N} V_{\text{pair}}(r_{ij})
$$
where
$$
V_{\text{pair}}(r_{ij}) = 4\epsilon_{\text{LJ}} \left[ \left(\frac{\sigma}{r_{ij}}\right)^{12} - \left(\frac{\sigma}{r_{ij}}\right)^6 \right]
$$
and $r_{ij} = \|r_i - r_j\|_2$.

**Target Distribution**:
$$
p_\infty(R) \propto \exp\left\{-\frac{V(R)}{\varepsilon^2}\right\}
$$

**SDE**:
$$
\begin{cases}
dX_t^{(i)} = -\tfrac{1}{2} \partial_{x_i} V(R_t) \, dt + \varepsilon \, dB_t^{(i,1)} \\
dY_t^{(i)} = -\tfrac{1}{2} \partial_{y_i} V(R_t) \, dt + \varepsilon \, dB_t^{(i,2)} \\
dZ_t^{(i)} = -\tfrac{1}{2} \partial_{z_i} V(R_t) \, dt + \varepsilon \, dB_t^{(i,3)}
\end{cases}, \quad i = 1, \ldots, N
$$

**Metrics**:
- **Pair-distance histogram $L^1$**: Compares empirical distribution of $\{r_{ij}\}$ to reference
- **Pair-distance histogram $L^2$**: RMSE of pair-distance distribution

**Hyperparameters** (`lennard_jones_potential.py`, example for $N=3$):
```python
sigma = 1.0          # LJ length scale
epsilon_LJ = 2.0     # LJ energy scale
eps = 0.1            # SDE noise scale
dt = 0.001           # Time step
T = 15.0             # Total time
N_particles = 3      # Number of particles
N_samples = 15000    # Ensemble size
alpha = 1.5          # FLMC tail index
lam = 1.2            # LSB-MC jump intensity
sigma_L = 0.8        # LSB-MC base jump magnitude
multipliers = [1.0, 1.8, 2.6]
pm = [0.70, 0.22, 0.08]
```

**LSB-MC Jump Details**:
- **High-dimensional isotropic jumps**: Jumps act on the full $3N$-dimensional configuration space as $(m_k \sigma_L) \cdot U$ where $U \sim \text{Uniform}(\mathbb{S}^{3N-1})$
- **Jump cap**: Individual jump magnitudes are clipped to prevent numerical explosion

---

### 6. Ten-Dimensional Separable Double-Well

**Mathematical Definition**:
$$
V(x) = \sum_{i=1}^{10} (x_i^2 - 1)^2
$$

**Target Distribution**:
$$
p_\infty(x) \propto \exp\left\{-\frac{2V(x)}{\epsilon^2}\right\}
$$

**SDE**:
$$
dX_t^{(i)} = -\partial_{x_i} V(X_t) \, dt + \epsilon \, dB_t^{(i)}, \quad i = 1, \ldots, 10
$$
with gradient
$$
\partial_{x_i} V(x) = 4 x_i (x_i^2 - 1)
$$

**Metrics**:
- **Sliced Wasserstein-2**: Projects onto 64 random 1D directions, computes exact $W_2$ per direction, averages
- **Orthant $L^1$ error**: Compares empirical orthant occupancy to uniform ($1/2^{10}$ per orthant)
- **Orthant $L^2$ error**: RMSE of orthant occupancy distribution

**Hyperparameters** (`high_dim.py`):
```python
sigma = 0.75         # Noise scale
dt = 0.005           # Time step
T = 10.0             # Total time
N = 20000            # Number of particles
dim = 10             # Dimension
alpha = 1.75         # FLMC tail index
lam = 0.8            # LSB-MC jump intensity
sigma_L = 1.0        # LSB-MC base jump magnitude
multipliers = [1.0, 1.8, 2.6]
pm = [0.70, 0.22, 0.08]
n_dir_score = 8      # Number of random directions for score approximation
n_theta = 7          # Quadrature points in θ ∈ [0,1]
```

**Fair Comparison Principle**:
Even though $V(x)$ is separable, **all methods treat it as a general 10D potential**:
- ULA/MALA use standard multivariate Brownian motion (coordinatewise independent by definition)
- **FLMC uses genuinely isotropic $\alpha$-stable vectors** $Z = R \cdot U$ where $U \sim \text{Uniform}(\mathbb{S}^9)$ and $R \sim S_\alpha^{1/\alpha}$ (not coordinatewise)
- **LSB-MC uses genuinely isotropic jumps** $(m_k \sigma_L) \cdot U$ with $U \sim \text{Uniform}(\mathbb{S}^9)$ (not coordinatewise)

This ensures no method exploits the separability structure, providing a fair benchmark of high-dimensional exploration.