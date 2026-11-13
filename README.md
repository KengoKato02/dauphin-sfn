# Learning Saddle-Free Newton with PyTorch

An educational implementation of **saddle-free Newton (SFN)** from Dauphin et al.,
[*Identifying and attacking the saddle point problem in high-dimensional non-convex
optimization*](https://arxiv.org/abs/1406.2572), using the supplied June 2014 v1 paper.

We want to understand the geometry, implement the method in small, inspectable
steps, and compare it with single-example SGD and minibatch SGD on MNIST and
CIFAR-10. Training runs in Python; Jupyter notebooks explain the ideas and plot
saved results.

## What the paper says

In high-dimensional non-convex optimization, slow learning can come from saddle
points surrounded by nearly flat regions. Near a saddle, some directions curve up
and others curve down. A small gradient can therefore mean a plateau near a saddle,
rather than arrival at a useful minimum.

The paper connects results from random Gaussian landscapes, random matrix theory,
and neural networks. It finds a relationship between the error at a critical point
and its **index**, the fraction of negative Hessian eigenvalues: higher-error
critical points tend to have more descending directions. Its neural network
experiments support this relationship qualitatively; the random-landscape results
are not a proof that every neural network has no bad local minima.

Gradient descent follows the right local direction but progresses slowly along
directions of weak curvature. Newton's method rescales those directions but can
move *toward* a saddle. SFN combines curvature-based rescaling with the descent
orientation of the gradient by replacing each Hessian eigenvalue with its absolute
value.

The experiments proceed from small MLPs on 10 x 10 versions of MNIST and CIFAR-10,
where the full Hessian can be computed, to an MNIST deep autoencoder and a Penn
Treebank recurrent network. The larger experiments use a Krylov approximation and
continue training after SGD has slowed. Those experiments concern optimization
performance; they do not establish a universal advantage in test accuracy or time.

## Project structure

```text
.
├── src/
│   ├── main.py                    # CLI, training, evaluation, saving metrics
│   └── models.py                  # Small tanh MLP shared by all algorithms
├── utils/
│   └── load.py                    # MNIST / CIFAR-10 loading and data splits
├── descent_algorithms/
│   ├── sgd.py                     # Single-example SGD
│   ├── minibatch_sgd.py            # Minibatch SGD
│   └── sfn.py                     # Exact, damped SFN reference
├── notebooks/
│   └── 01_saddles_and_training.ipynb
├── tests/                         # Numerical and offline integration checks
├── data/                          # Downloaded datasets; ignored by Git
├── results/                       # Per-run CSV / JSON artifacts; ignored by Git
└── pyproject.toml                 # Dependencies, package metadata, tooling
```

`data/` and `results/` are created when needed. The root utility/algorithm packages
and `src/main.py` follow the layout above and are importable through an editable
installation. There are no dataset downloads or training side effects on import.

## Environment

Use Python 3.12; `.python-version` records 3.12.10. The PyTorch/torchvision pins
preserve the versions already installed in this project's virtual environment.
`pyproject.toml` is the single project configuration file: it declares dependencies,
how the Python packages are installed, and lint/test settings. The `dev` extra adds
Jupyter, ipykernel, and development tools. This is not a fully locked environment.

From the project root, activate the existing virtual environment and install:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On a fresh checkout without `.venv`, first run `python3.12 -m venv .venv`.
The editable installation (`-e`) keeps imports connected to the source files as
you edit them.

For runtime dependencies without notebook/development tools, use
`python -m pip install -e .` instead.

Register a named Jupyter kernel inside the activated `.venv`:

```bash
python -m ipykernel install --sys-prefix \
  --name dauphin-sfn \
  --display-name "Python (dauphin-sfn)"
jupyter lab
```

`--sys-prefix` stores the kernel registration inside `.venv`. In Jupyter's kernel
picker, select **Python (dauphin-sfn)**. The kernel uses this environment's Python
and dependencies. The notebook locates the repository whether Jupyter starts in
the project root or in `notebooks/`.

Useful project commands:

```bash
python -m src.main --help
python -m pytest
ruff check .
ruff format --check .
```

## First experiments

Small subsets make it quick to check the complete workflow. The first command for
each dataset needs `--download`; later runs read cached files in `data/`.

```bash
# Minibatch SGD on a small MNIST subset
python -m src.main --dataset mnist --algorithm minibatch_sgd --download \
  --epochs 3 --train-limit 512 --eval-limit 128 --batch-size 32 --lr 0.05

# Single-example SGD: the effective batch size is always one
python -m src.main --dataset mnist --algorithm sgd \
  --epochs 3 --train-limit 512 --eval-limit 128 --lr 0.01

# Exact SFN: start with a very small problem and a single epoch
python -m src.main --dataset mnist --algorithm sfn \
  --epochs 1 --train-limit 64 --eval-limit 64 --batch-size 32 --lr 0.1 --damping 0.1

# CIFAR-10 retains RGB channels; the same interface handles both datasets
python -m src.main --dataset cifar10 --algorithm minibatch_sgd --download \
  --epochs 3 --train-limit 512 --eval-limit 128 --batch-size 32 --lr 0.05
```

Defaults use a five-unit tanh hidden layer, 10 x 10 images, CPU, and float64.
The loader can also return native-resolution images with `image_size=None` when
called from Python. The CLI uses `--image-size` to select the resized resolution.
Without subset limits the full training, validation, and test splits are used.

The official training set is split into 90% training and 10% validation with a
fixed seed. Optional subset limits are applied after that split. Images are mapped
to [-1, 1] with fixed normalization, without augmentation. The official test set
is evaluated after training; use validation metrics for choosing hyperparameters.

Every run creates a unique `results/<dataset>_<algorithm>_<timestamp>_<id>/` with:

- `config.json`: arguments, effective batch size, split sizes, parameter count,
  and torch/torchvision versions.
- `metrics.csv`: epoch-zero baseline and per-epoch training/validation loss and
  accuracy, cumulative updates, examples seen, and training seconds.
- `test_metrics.json`: final test loss and accuracy.

Training metrics are evaluated at the current weights after each epoch, not
averaged over changing weights during optimization. Accuracy is stored as a
fraction between 0 and 1. Training time includes data loading and updates, excludes
evaluation, and synchronizes accelerator work. No checkpoints are saved yet.

Matching seeds reproduce initialization, splitting, and shuffle streams in this
CPU setup; exact reproducibility across hardware/PyTorch versions is not promised.
Compare runs with matching data, model, preprocessing, and seeds. SGD and
minibatch methods perform different numbers of updates per epoch, and one SFN
update costs much more than one SGD update. Plot loss against epochs, updates,
and elapsed training time before drawing conclusions. The example learning rates
are starting points, not tuned results.

## The mathematics

### Loss, gradient, and Hessian

For parameters $\theta \in \mathbb{R}^P$ and $N$ training examples, define

$$
f(\theta) = \frac{1}{N}\sum_{i=1}^{N}\ell(h_\theta(x_i), y_i),
\qquad g = \nabla f(\theta),
\qquad H = \nabla^2 f(\theta).
$$

Here $h_\theta$ produces class logits and $\ell$ is cross entropy. For a minibatch
$B$, replace $f$ by its mean loss $f_B$. Our SFN reference computes both $g_B$
and $H_B$ on the **same batch**, so it is exact for that batch's loss, rather than
for the full-dataset objective.

At a critical point $g=0$, a nonsingular symmetric Hessian classifies the point:
all positive eigenvalues indicate a local minimum; all negative indicate a local
maximum; mixed signs indicate a saddle. If some eigenvalues are zero, second-order
information alone is insufficient. Appendix A also discusses degenerate saddles
and flat sets of critical points.

### SGD and minibatch SGD

Both use

$$
g_B = \frac{1}{|B|}\sum_{i\in B}\nabla_\theta\ell(h_\theta(x_i),y_i),
\qquad \theta_{t+1}=\theta_t-\eta g_B.
$$

In this project **SGD** means $|B|=1$; **minibatch SGD** means $|B|>1$.
They are the same update with different batch sizes. Each has its own file, using
PyTorch's SGD implementation without momentum or weight decay. `main.py` enforces
the selected batch size; the optimizer itself does not split the data.

### Why Newton can approach a saddle

The local second-order model is

$$
f(\theta+s) \approx f(\theta)+g^Ts+\frac12s^THs,
\qquad s_{\mathrm{Newton}}=-H^{-1}g.
$$

Write $H=Q\Lambda Q^T$, with orthonormal eigenvectors $q_i$ and eigenvalues
$\lambda_i$. Newton rescales each gradient component by $1/\lambda_i$. A negative
eigenvalue reverses the gradient's descent direction in that eigendirection.
Near a nondegenerate quadratic saddle, Newton can therefore converge to the saddle.

For example, $f(x,y)=5x^2-y^2$ has $g=(10x,-2y)^T$ and
$H=\operatorname{diag}(10,-2)$. Newton gives $s=(-x,-y)^T$, taking a full step
directly to the saddle at the origin. Gradient descent moves away along $y$, but
can be slow when the magnitude of negative curvature is small.

### The saddle-free step

Define the matrix absolute value spectrally:

$$
|H|=Q|\Lambda|Q^T,
\qquad |\Lambda|_{ii}=|\lambda_i|.
$$

This is **not** the elementwise absolute value of $H$. The ideal SFN step from
Eq. (5), written using column vectors, is

$$
s_{\mathrm{SFN}}=-\eta|H|^{-1}g.
$$

Our damped reference uses $\mu>0$ to handle small or zero eigenvalues:

$$
s=-\eta(|H|+\mu I)^{-1}g
  =-\eta\sum_i\frac{q_i^Tg}{|\lambda_i|+\mu}q_i.
$$

We compute this in the eigenbasis with `torch.linalg.eigh`, without constructing
an explicit inverse. The magnitude rescales each component, while its descent
orientation is preserved. For the undamped quadratic example above, SFN gives
$s=\eta(-x,+y)^T$: toward the minimum in $x$ and away from the saddle in $y$.
When $H$ is positive definite, undamped SFN and Newton agree.

With positive damping and $g\ne0$, $g^Ts<0$, so this is a descent *direction*.
A fixed, finite step can still increase a nonlinear loss. Also, when $g=0$
exactly, the SFN step is zero: it cannot spontaneously leave an exact stationary
point. Flat or degenerate directions may require more than second-order analysis.

### Where the step comes from

Sections 5-6 motivate keeping the first-order objective reliable. For symmetric
$H$, the second-order correction has the upper bound

$$
\left|\frac12s^THs\right|\le\frac12s^T|H|s.
$$

The inequality follows by expanding $s$ in the Hessian eigenbasis and applying
the triangle inequality. The paper then solves a linear objective subject to a
curvature-shaped trust region, absorbing the factor of two into its radius:

$$
\min_s g^Ts \quad\text{subject to}\quad s^T|H|s\le\Delta.
$$

If $|H|$ is positive definite and $g\ne0$, the Lagrangian gives
$g+2\nu|H|s=0$, hence $s=-\eta|H|^{-1}g$ for a positive scalar $\eta$ determined
by the trust-region radius. Curvature defines the region's shape rather than a
quadratic objective to minimize. Damping makes the metric positive definite when
the Hessian is singular. Our first implementation uses fixed $\eta$ and $\mu$;
it does not solve for a trust-region radius or adapt damping at each update.

### Scaling beyond the exact Hessian

A dense Hessian needs $O(P^2)$ storage, and its full eigendecomposition costs
$O(P^3)$. The reference differentiates each gradient component to form the Hessian
and rejects models over 2,000 trainable parameters before allocating it. The default
MLP has 565 parameters on resized MNIST and 1,565 on resized RGB CIFAR-10. This is
for learning and small checks, not large models or efficient training.

Algorithm 1 and Appendix E instead build a $k$-dimensional Krylov/Lanczos basis
$V\in\mathbb{R}^{P\times k}$ with $k\ll P$ using Hessian-vector products:

$$
Hv=\nabla_\theta\big(g^Tv\big),\qquad
\widehat g=V^Tg,\qquad \widehat H=V^THV,
$$

$$
s\approx-\eta V(|\widehat H|+\mu I)^{-1}\widehat g.
$$

This keeps the spectral calculation in a smaller space. The paper also includes
the previous update direction in the basis and takes multiple inner steps while
reusing projected curvature. Appendix F selects damping per update from a
candidate grid. These are future steps here; the dense reference does not yet
reproduce the paper's approximate algorithm or tuned benchmarks.

## A learning path

1. Run the notebook's two-dimensional saddle example and inspect how Newton,
   gradient descent, and SFN move (Sections 4 and 6; Appendices A-B).
2. Read `sfn.py` alongside the eigenbasis formula and check the quadratic tests.
3. Run tiny MNIST experiments, then vary damping, learning rate, and batch size.
4. Add Hessian eigenvalue diagnostics and investigate the error/index relationship
   in Figure 1 and Appendix C. A small gradient alone does not identify a saddle.
5. Add Hessian-vector products, a tested Lanczos basis, and the projected SFN step
   from Algorithm 1 / Appendix E before attempting larger networks.
6. Add adaptive damping, a damped-Newton baseline, multiple seeds, and comparable
   hyperparameter searches before attempting to reproduce Figure 3. The paper's
   Section 7 / Appendix F describe the experimental protocol.

This first pass provides working loaders, SGD baselines, a dense SFN reference,
metric logging, and a starter notebook. The scalable method, deep autoencoder,
RNN experiments, and paper-level reproduction remain deliberate next steps.
