# LSB-MC: Lévy-Score-Based Monte Carlo for Boltzmann Sampling

## Overview

Sampling from Boltzmann distributions of the form

$$
p_\infty(x) = Z^{-1} \exp\left(-\frac{2V(x)}{\sigma^2}\right)
$$

via overdamped Langevin dynamics becomes prohibitively slow when the potential $V$ features multiple deep wells separated by high energy barriers. LSB-MC addresses this by augmenting the standard diffusion process with **compound Poisson jumps** and a **stationary Lévy-score correction**, enabling macroscopic spatial transitions while preserving the target invariant measure.

## Sampling Methods

This repository implements four sampling algorithms targeting the same Boltzmann distribution. All methods are compared on identical test problems to assess convergence rates and exploration efficiency.

---

### 1. ULA (Unadjusted Langevin Algorithm)

**Target Distribution**:

$$
p_\infty(x) = Z^{-1} \exp\left(-\frac{2V(x)}{\sigma^2}\right), \quad x \in \mathbb{R}^d
$$

**Continuous-Time SDE**:

$$
dX_t = -\nabla V(X_t)  dt + \sigma  dB_t
$$

where $B_t$ is standard Brownian motion and $\sigma > 0$ is the noise intensity.

**Discrete-Time Update** (Euler-Maruyama):

$$
X_{n+1} = X_n + \frac{dt \cdot (-\nabla V(X_n))}{1 + dt \|\nabla V(X_n)\|} + \sigma \sqrt{dt}  Z_n, \quad Z_n \sim \mathcal{N}(0, I)
$$

The denominator $1 + dt \|\nabla V(X_n)\|$ implements **taming** for numerical stability when gradients are large.

---

### 2. MALA (Metropolis-Adjusted Langevin Algorithm)

**Target Distribution**: 

$$
p_\infty(x) = Z^{-1} \exp\left(-\frac{2V(x)}{\sigma^2}\right), \quad x \in \mathbb{R}^d
$$

**Proposal Distribution**:

$$
Y = X + \frac{1}{2} dt  \nabla \log p_\infty(X) + \sqrt{dt}  Z, \quad Z \sim \mathcal{N}(0, I)
$$
where $\nabla \log p_\infty(x) = -2\nabla V(x)/\sigma^2$.

**Acceptance Probability**:

$$
\alpha(X, Y) = \min\left(1, \frac{p_\infty(Y)  q(X \mid Y)}{p_\infty(X)  q(Y \mid X)}\right)
$$

where $q(Y \mid X)$ is the Gaussian proposal kernel:

$$
q(Y \mid X) \propto \exp\left(-\frac{\|Y - X - \tfrac{1}{2}dt  \nabla \log p_\infty(X)\|^2}{2dt}\right)
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
p_\infty(x) = Z^{-1} \exp\left(-\frac{2V(x)}{\sigma^2}\right), \quad x \in \mathbb{R}^d
$$

**Continuous-Time SDE**:

$$
dX_t = -c_\alpha \nabla V(X_t)  dt + \sigma  dt^{1/\alpha}  dL_t^\alpha
$$

where:
- $L_t^\alpha$ is a symmetric $\alpha$-stable Lévy process with tail index $\alpha \in (1, 2]$
- $c_\alpha = \Gamma(\alpha - 1) / \Gamma(\alpha/2)^2$ is a normalization constant

**Discrete-Time Update**:

$$
X_{n+1} = X_n + \frac{dt \cdot (-c_\alpha \nabla V(X_n))}{1 + dt \|c_\alpha \nabla V(X_n)\|} + \sigma  dt^{1/\alpha}  Z_n
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
p_\infty(x) = Z^{-1} \exp\left(-\frac{2V(x)}{\sigma^2}\right), \quad x \in \mathbb{R}^d
$$

**Continuous-Time SDE**:

$$
dZ_t = \left(-\nabla V(Z_{t-}) + S_L^s(Z_{t-})\right) dt + \sigma  dB_t + dL_t
$$

where:
- $S_L^s(x)$ is the **stationary Lévy-score correction**:

$$
S_L^s(x) = -\int_0^1 \int_{\mathbb{R}^d \setminus \{0\}} r \exp\left(-\frac{2(V(x - \theta r) - V(x))}{\sigma^2}\right) \nu(dr)  d\theta
$$

- $\nu$ is the Lévy measure governing the jump law
- $L_t$ is a pure-jump Lévy process with measure $\nu$

**Compound Poisson Jump Law**:
In this implementation, $\nu = \lambda  \nu_J$ where:
- $\lambda > 0$ is the **jump intensity** (expected number of jumps per unit time)
- $\nu_J$ is a **discrete mixture of isotropic jumps**:

$$
\nu_J = \sum_{k=1}^K p_k  \delta_{m_k \sigma_L}
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
p_\infty(x) \propto \exp\left(-\frac{2V(x)}{\sigma^2}\right)
$$

**SDE**:

$$
dX_t = (X_t - X_t^3)  dt + \epsilon  dB_t
$$

where $\epsilon = \sigma$ under the global convention.

**Metrics**:
- **Wasserstein-2 distance**
- **MMD**
- **EMC**
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
p_\infty(x, y) \propto \exp\left(-\frac{2V(x, y)}{\sigma^2}\right)
$$

**SDE**:

$$
\begin{cases}
dX_t = -\partial_x V(X_t, Y_t)  dt + \sqrt{\epsilon}  dB_t^{(1)} \\
dY_t = -\partial_y V(X_t, Y_t)  dt + \sqrt{\epsilon}  dB_t^{(2)}
\end{cases}
$$

where $\sigma = \sqrt{\epsilon}$ (matches global convention).

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
p_\infty(x, y) \propto \exp\left(-\frac{V(x, y)}{\epsilon^2}\right)
$$

**SDE**:

$$
\begin{cases}
dX_t = -\tfrac{1}{2} \partial_x V(X_t, Y_t)  dt + \epsilon  dB_t^{(1)} \\
dY_t = -\tfrac{1}{2} \partial_y V(X_t, Y_t)  dt + \epsilon  dB_t^{(2)}
\end{cases}
$$


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
p_\infty(x, y) \propto \exp\left(-\frac{V(x, y)}{\epsilon^2}\right)
$$

**SDE**:

$$
\begin{cases}
dX_t = -\tfrac{1}{2} \partial_x V(X_t, Y_t) dt + \epsilon dB_t^{(1)} \\
dY_t = -\tfrac{1}{2} \partial_y V(X_t, Y_t) dt + \epsilon dB_t^{(2)}
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

### 5. Lennard-Jones Potential (7 atom model on 2D)

Adapted from [100 Years of the Lennard-Jones Potential](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00135).

**Mathematical Definition**:

Let $R = (r_1, \ldots, r_7)$ with $r_i \in \mathbb{R}^2$, and define the pair distances

$$
r_{ij} = \|r_i - r_j\|, \quad 1 \leq i < j \leq 7
$$

The 7-atom Lennard-Jones energy is

$$
V(R) = 4 \sum_{1 \leq i < j \leq 7} \left( r_{ij}^{-12} - r_{ij}^{-6} \right)
$$

The free cluster is translation-invariant, so the implementation works in the
center-of-mass-free gauge

$$
\sum_{i=1}^7 r_i = 0
$$

throughout the simulation.

**Target Distribution**:

$$
p_\infty(R) \propto \exp\left(-\frac{V(R)}{T^\star}\right)
= \exp\left(-\frac{2V(R)}{\sigma^2}\right), \quad \sigma = \sqrt{2T^\star}
$$

**SDE**:

$$
dr_t^{(i)} = -\nabla_{r_i} V(R_t)  dt + \sqrt{2T^\star}  dB_t^{(i)}, \quad i = 1, \ldots, 7
$$

where each $B_t^{(i)}$ is a 2D Brownian motion. In the code, states are recentered
after each update to remain in the center-of-mass-free subspace.

**Model Notes**:
- The ambient coordinate dimension is $7 \times 2 = 14$
- The landscape is highly nonconvex, with several compact cluster minima
- The active benchmark compares samplers through the sorted pair-distance descriptor in $\mathbb{R}^{21}$, which is invariant to translation, rotation, and atom relabeling

**Hyperparameters**:
```python
T_star = 0.05        # Reduced temperature
dt = 1.0e-3          # Time step
total_time = 1.0     # Total simulation time
n_samples = 256      # Number of particles / parallel chains
alpha = 1.5          # FLMC tail index
lam = 1.5            # LSB-MC jump intensity
sigma_L = 0.60       # LSB-MC base jump magnitude
jump_multipliers = [1.0, 1.7, 2.4]
jump_weights = [0.70, 0.22, 0.08]
n_dir_score = 4      # Random directions for score estimation
n_theta = 5          # Quadrature points in theta
```

---

### 6. Ten-Dimensional Separable Double-Well

**Mathematical Definition**:

$$
V(x) = \sum_{i=1}^{10} (x_i^2 - 1)^2
$$

**Target Distribution**:

$$
p_\infty(x) \propto \exp\left(-\frac{2V(x)}{\epsilon^2}\right)
$$

**SDE**:

$$
dX_t^{(i)} = -\partial_{x_i} V(X_t)  dt + \epsilon  dB_t^{(i)}, \quad i = 1, \ldots, 10
$$

with gradient

$$
\partial_{x_i} V(x) = 4 x_i (x_i^2 - 1)
$$

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

## Evaluation Metrics

Adapted from [Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling](https://arxiv.org/abs/2406.07423) (ICML 2024).

**Convention.** Let $\mu_t^N$ denote the empirical distribution of the $N$-particle ensemble at time $t$, and let $p_\infty$ denote the target Boltzmann distribution. Lower values of Sinkhorn and MMD indicate better convergence to $p_\infty$; higher values of EMC indicate better exploration across metastable basins.

---

### 1. Sinkhorn Divergence

**What it measures.** A debiased, regularised approximation of the squared **Wasserstein-2 distance** $W_2^2(\mu, \nu)$. The Wasserstein-2 distance metrises weak convergence of probability measures and is sensitive to the *geometry* of the space: a distribution that places mass in the wrong location incurs a cost proportional to the squared displacement, regardless of the shape of the density.

**Definition.** For two probability measures $\mu$ and $\nu$ on $\mathbb{R}^d$, the squared Wasserstein-2 distance is

$$
W_2^2(\mu, \nu) = \inf_{\pi \in \Pi(\mu, \nu)} \int \|x - y\|^2   d\pi(x, y)
$$

where $\Pi(\mu, \nu)$ is the set of all couplings with marginals $\mu$ and $\nu$. Computing this directly is expensive. Introducing an entropy regulariser $\varepsilon > 0$ yields the **entropic transport plan**

$$
\pi^\varepsilon(\mu,\nu) = \text*{argmin}_{\pi \in \Pi(\mu,\nu)} \left[\int \|x - y\|^2   d\pi(x,y) - \varepsilon   H(\pi)\right]
$$

where $H(\pi) = -\int \log \frac{d\pi}{d({\mu \otimes \nu})}   d\pi$ is the relative entropy of $\pi$ with respect to the product measure. The **Sinkhorn divergence** corrects for the $O(\varepsilon)$ bias introduced by regularisation:

$$
S_\varepsilon(\mu, \nu) = \langle C, \pi^\varepsilon(\mu,\nu)\rangle - \frac{1}{2}\langle C, \pi^\varepsilon(\mu,\mu)\rangle - \frac{1}{2}\langle C, \pi^\varepsilon(\nu,\nu)\rangle
$$

where $C(x,y) = \|x-y\|^2$. The two self-transport terms ensure $S_\varepsilon(\mu,\mu) = 0$ exactly and $S_\varepsilon(\mu,\nu) \to W_2^2(\mu,\nu)$ as $\varepsilon \to 0$.

**Practical computation.** Given empirical samples, $\pi^\varepsilon$ is computed via the **Sinkhorn–Knopp** algorithm on the kernel matrix $K_{ij} = \exp(-\|x_i - y_j\|^2/\varepsilon)$. A stabilised log-domain variant is used when $\varepsilon$ is small.

**Interpretation.**
- $S_\varepsilon(\mu, \nu) = 0$ if and only if $\mu = \nu$.
- A large value means that transforming $\mu_t^N$ into $p_\infty$ requires moving mass over large distances in the sample space — a symptom of metastability, where the ensemble is stuck near its starting basin and has not spread to cover the full support of $p_\infty$.

**Lower is better.**

---

### 2. Maximum Mean Discrepancy (MMD)

**What it measures.** The distance between $\mu$ and $\nu$ in a **reproducing kernel Hilbert space** (RKHS). MMD is sensitive to differences in the *moments* of the two distributions, at all length scales determined by the kernel, without requiring an optimisation problem.

**Definition.** For a symmetric, positive-definite kernel $k : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ with associated feature map $\phi$ and RKHS $\mathcal{H}$, the squared MMD is the squared distance between the **kernel mean embeddings** $m_\mu = \mathbb{E}_{x \sim \mu}[\phi(x)]$ and $m_\nu = \mathbb{E}_{y \sim \nu}[\phi(y)]$:

$$
\text{MMD}^2(\mu, \nu) = \|m_\mu - m_\nu\|_{\mathcal{H}}^2 = \mathbb{E}_{x,x' \sim \mu}[k(x,x')] - 2 \mathbb{E}_{\substack{x \sim \mu \\ y \sim \nu}}[k(x,y)] + \mathbb{E}_{y,y' \sim \nu}[k(y,y')]
$$

If $k$ is a **characteristic kernel** (e.g. any Gaussian), then $\text{MMD}(\mu,\nu) = 0$ if and only if $\mu = \nu$.

**Kernel choice.** A multi-scale mixture of $M$ Gaussian kernels with geometrically spaced bandwidths is used:

$$
k(x, y) = \frac{1}{M}\sum_{m=1}^{M} \exp\left(-\frac{\|x - y\|^2}{2 h_m^2}\right), \quad h_m = h_0 \cdot \rho^{m-1}
$$

The mixture is sensitive to differences at multiple length scales simultaneously, avoiding the bandwidth-selection problem inherent to a single-kernel estimator.

**Unbiased estimator.** Given $n$ samples $\{x_i\}$ from $\mu$ and $m$ samples $\{y_j\}$ from $\nu$:

$$
\widehat{\text{MMD}}^2 = \frac{1}{n(n-1)}\sum_{i \neq j} k(x_i, x_j) - \frac{2}{nm}\sum_{i,j} k(x_i, y_j) + \frac{1}{m(m-1)}\sum_{i \neq j} k(y_i, y_j)
$$

In practice $\nu = p_\infty$ is represented by a large precomputed reference sample.

**Interpretation.**
- A sampler whose distribution matches $p_\infty$ in mean, variance, and higher moments will have small MMD even if individual samples are spread differently from the Sinkhorn perspective.
- Compared to the Sinkhorn divergence, MMD is cheaper to compute (no optimisation) but does not have a direct geometric interpretation in terms of mass displacement.

**Lower is better.**

---

### 3. Entropic Mode Coverage (EMC)

**What it measures.** Whether the sampler **visits all metastable basins** and how uniformly it distributes mass across them. This is an *exploration* metric — it is orthogonal to distribution fidelity. A sampler can achieve low Sinkhorn/MMD by slowly filling the correct marginal from one basin; EMC detects whether it is actively crossing barriers to explore the full support.

**Setup.** Assume the target $p_\infty$ has $K$ identifiable metastable modes with representative configurations $\{m_1, \ldots, m_K\}$ (local minima of $V$, or centres of mass of the $K$ modes). Each sample $x_i$ from the ensemble is **assigned** to one of the $K$ modes or marked unassigned:

$$
\ell_i = \begin{cases} \arg\min_{k}   d(x_i, m_k) & \text{if } \min_k d(x_i, m_k) \leq r^* \\ -1 & \text{(unassigned)} \end{cases}
$$

where $d(\cdot, \cdot)$ is a suitable dissimilarity in sample space and $r^*$ is an assignment radius. Samples that do not belong to any identifiable basin (e.g. dissociated configurations, saddle-point regions) are left unassigned.

**Definition.** Let $f = n_{\text{assigned}} / n$ be the fraction of assigned samples and $p_k = n_k / n_{\text{assigned}}$ the empirical occupancy of mode $k$ among assigned samples. Define the Shannon entropy of the conditional occupancy:

$$
H = -\sum_{k=1}^{K} p_k \log p_k \qquad (\text{sum over nonzero } p_k)
$$

The **Entropic Mode Coverage** is

$$
\boxed{\text{EMC} = f \cdot \frac{e^H}{K} \in [0, 1]}
$$

**Decomposition.** EMC factors into two interpretable components:

$$
\text{EMC} = \underbrace{f}_{\substack{\text{assigned} \\ \text{fraction}}} \times \underbrace{\frac{e^H}{K}}_{\substack{\text{conditional} \\ \text{mode entropy}}}
$$

- The **conditional mode entropy** $e^H/K \in [1/K,  1]$ measures how uniformly assigned samples spread across the $K$ modes. It equals $1$ when all modes are equally occupied and $1/K$ when all assigned mass collapses to a single mode. Since $e^H$ is the perplexity of the distribution $\{p_k\}$, dividing by $K$ normalises it to the unit interval.
- The **assigned fraction** $f \in [0,1]$ penalises methods that scatter particles into inter-basin regions. A sampler with perfectly uniform mode coverage but $f \ll 1$ is not genuinely exploring the target basins; multiplying by $f$ reflects this.

**Boundary cases.**

| Scenario | $f$ | $e^H/K$ | EMC |
|---|---|---|---|
| All samples in one mode, all assigned | $1$ | $1/K$ | $1/K$ |
| All samples assigned, perfectly uniform | $1$ | $1$ | $1$ |
| All samples unassigned | $0$ | — | $0$ |
| Half assigned, uniform over $K$ modes | $1/2$ | $1$ | $1/2$ |

**Interpretation.**
- $\text{EMC} = 1$ requires both that every sample is assigned to a recognised basin *and* that the $K$ modes are equally occupied.
- $\text{EMC} = 1/K$ is the minimum non-trivial value, achieved when all assigned samples concentrate in a single mode.
- EMC does not assess within-basin density accuracy. A sampler can achieve high EMC while still having the wrong within-well shape; use Sinkhorn or MMD for distributional fidelity.
- EMC is most informative in the **transient phase** before equilibration: it directly measures barrier-crossing ability rather than long-time stationarity.

**Higher is better.**

---
