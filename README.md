# Evolving HEP Analyses

tldr: *By removing hard binning and coupling all stages, we transform analysis
into a continuous, optimizable process. This does not throw away physics -- it
tightly weaves physics and statistical inference together. With gradients
flowing from the likelihood back to the detector parameters, we can
systematically tune calibrations and designs to maximize discovery potential,
all while keeping models faithful and validated. This represents a new analysis
paradigm for the next generation of experiments*

## Overview

This end-to-end HEP analysis pipeline is a minimal, teaching-oriented C++
implementation. It is not meant to be a high-performance or fully realistic
simulation, but rather a clear illustration of how a modern analysis chain --
from theory parameters to final likelihood -- can be cast as a single
differentiable graph. Each stage of the pipeline (generation, simulation,
digitization, reconstruction, analysis, inference) is represented by a compact
function that mimics a real subsystem. The key idea is that we can propagate
gradients backward through all stages to see how changes in fundamental
parameters or calibration constants would affect the final likelihood. Recent
work has shown that such differentiable pipelines allow "end-to-end tuning of
analysis parameters," greatly improving sensitivity.

In formulae, we treat the pipeline as a map

$$(\theta, g) \longrightarrow data \longrightarrow \{x_{obs}\} \longrightarrow L(\{x\} \mid \theta, g),$$

where $\theta$ are physics parameters (e.g. particle masses) and $g$ represents
detector calibration parameters (e.g. gains). We then compute gradients

$$\frac{\partial L}{\partial \theta}, \frac{\partial L}{\partial g},$$


by applying the chain rule through every module. This means we can optimize or
tune $\theta,g$ directly (e.g. via gradient descent) instead of doing
brute-force scans. Conceptually, the analysis is no longer "final" or static --
it becomes integrated with the whole simulation chain. This hybrid approach
preserves the familiar modular workflow (calibration, reconstruction, inference)
while enabling gradients to flow through it.

## Mapping Toy Code to Real Experiments

Below is a summary of each toy function and its real-world counterpart. The
table shows the stage, the toy function (role), and the production equivalent
subsystem.

| Stage | Toy Function / Role | Real Experiment Example |
| :--- | :--- | :--- |
| Shower Simulation | `simulate_shower`: parametric EM shower model. Takes a truth electron and generates calorimeter hits across layers, with Gaussian transverse spread and exponential longitudinal profile. | Event generation (e.g. MadGraph/Pythia), followed by detailed detector simulation (GEANT4 or HepEmShow) that models particle showers. Our toy is a very coarse surrogate of the full GEANT4 shower. (In fact, recent studies are applying automatic differentiation to GEANT4 itself |
| Digitization & Readout | `digitize_hits`: maps continuous hits into discrete calorimeter cells. It sums energies per cell, then applies a gain calibration constant $g$ and adds Gaussian electronics noise. Outputs ADC counts and timestamps, keeping the original hit energies for the backward pass. | Electronics simulation (CMS `EcalElectronicsSim`, ATLAS `LAr` digitization). Uses calibration constants (gains, pedestals) from the conditions database. This stage is where detector calibration parameters like $g$ naturally enter. |
| Reconstruction | `reconstruct`: groups digis into physics objects. In this toy, each digi simply becomes one `ElectronCandidate` with energy = ADC. We keep a link (`cellIds`) to the contributing cells. | Object reconstruction (CMS `reco::GsfElectron`, ATLAS `topoclusters/electron` reco). In reality, this involves clustering many cells, track-cluster matching, ID algorithms, and so on. |
| Observables & Likelihood | `invariant_mass` & `likelihood`: take two electrons, compute their invariant mass $m$, and evaluate a simple Gaussian likelihood $L(m_Z,\sigma)$. This mimics a $Z\to ee$ analysis. | TODO |
| Gradient Backward Pass | `backward_gain`: manually computes $\partial L/\partial g$ by backpropagating through the likelihood, invariant-mass, and digitization. We calculate $\partial L/\partial m$, then $\partial m/\partial E_i$, and use $E_i = g\cdot(HitEnergy)$ to get $dL/dg$. | Conceptually, this is akin to adjoint methods or autodiff-enabled calibration: computing how the final likelihood changes if calibration constants or detector parameters are tweaked. In a full system one might use operator-overloaded AD (e.g. differentiable GEANT4) or score-based estimators. |

Each row above corresponds to a real subsystem in an experiment (CMS, ATLAS,
etc.), making the mapping explicit. For example, our simulate_shower toy roughly
stands in for a GEANT4 shower simulation -- though it is a very simplified
physics-inspired model. (Crucially, our parametric shower respects basic physics
laws like energy conservation.) In real projects, teams are exploring ways to
make full simulations differentiable, but here we use a hand-crafted surrogate
for clarity.

## Toy Pipeline Components

Below we outline each toy component. The goal is pedagogical clarity, so we
mention the real analog and point out where the differentiability comes in:

* **Simulation** (`simulate_shower`): This function takes a *Particle* with
  momentum $\left(p_x,p_y,p_z,E \right)$ and simulates a stack of calorimeter
  layers. In each layer it generates several random hits with positions $(x,y)$
  smeared by a Gaussian (`sigma_xy`) around the electron’s trajectory, and
  energies following an exponential decay in depth. Each hit is recorded as a
  `CaloHit` with `(layer,x,y,energy,time)`. This is a toy model for the
  electromagnetic shower. In a real experiment, one would use a full event
  generator (e.g. MadGraph+Pythia to create particles) and a detailed detector
  simulation (GEANT4 or HepEm) to simulate hundreds of interactions in each
  layer. Our model is much simpler, but it is **differentiable**: the hit
  energies depend explicitly on the input particle energy. (We use a fixed RNG
  seed for reproducibility in teaching.) Notably, there are research efforts to
  differentiate actual GEANT4 simulations, which underscores that one can in
  principle backpropagate through the true physics model too.

* **Digitization** (`digitize_hits`): This function mimics the electronics
  readout. It takes the calorimeter hits and assigns each $(x,y)$ to a cell
  index (discretizing by `cell_size`). It sums energies per cell, then applies a
  gain factor $g$ (the calibration parameter) and adds Gaussian noise
  (`noise_sigma`). It outputs a list of `Digi` objects with `(cellId, adc, time,
  hit_energy)`. The `hit_energy` field is a hidden variable used only for
  computing gradients. In production, this step corresponds to the front-end
  electronics simulation: mapping energy deposits to ADC counts using
  calibration constants (gains, pedestals) from a database (e.g. CMS Ecal, ATLAS
  LAr). Calibration parameters live here; our toy makes $adc = g \times
  (\text{true energy}) + \text{noise}$. The backward pass will differentiate
  through this simple linear relation to update $g$.

* **Reconstruction** (`reconstruct`): Groups digis into physics objects. In this
  toy version, we do the simplest thing: each digi with nonzero ADC becomes one
  `ElectronCandidate` with that energy (and zero momentum components aside from
  energy), and we record the associated cell ID. (Hence we effectively assume
  each cell is one tiny electron object.) In reality, reconstruction is much
  more complex: e.g., CMS clustering (`SuperCluster`), Gaussian-sum filters for
  electrons, track matching, shower-shape analysis, etc. The important point is
  that we preserve the link back to which cells contributed to the energy. The
  reconstruction stage in real life would also involve many non-linear
  algorithms, but our toy simply demonstrates how one might propagate a gradient
  through reconstruction if needed (e.g. if cluster calibration or
  identification scores were differentiable).

* **Physics Analysis** (`invariant_mass`, `likelihood`): From the reconstructed
  objects we build high-level observables. Here we take the first two electrons
  and compute their invariant mass $m = \sqrt{(E\_1 + E\_2)^2 - |\vec{p}\_1 + \vec{p}\_2|^2}$.
  We then evaluate a Gaussian likelihood

  $$L = \frac{1}{\sigma \sqrt{2\pi}} \exp\left[ - \frac{(m - m_Z)^2}{2\sigma^2} \right],$$

  with $m_Z=91.1876$ GeV and $\sigma=2$ GeV by default. This mimics searching for a
  $Z\to ee$ peak. In real analyses, one would fill a $m_{ee}$ histogram and fit
  it (e.g. with RooFit/RooStats) or build an unbinned likelihood. Modern
  frameworks like pyhf implement these statistical models with support for
  tensor backends and automatic differentiation indico.cern.ch , so the
  statistical inference itself can be made differentiable. Our toy likelihood is
  simple, but it illustrates how we can compute $\partial L/\partial m$
  analytically for the backward pass.

* **Backward Pass (calibration gradient)**: The function `backward_gain` manually
  computes $\partial L/\partial g$ using the chain rule. First it computes
  $dL/dm$ from the Gaussian since $L=\exp[-(m-m_Z)^2/(2\sigma^2)]/(\sigma\sqrt{2\pi})$. Then it computes how
  the invariant mass changes with respect to each electron’s energy: $dm/dE_i =
  (E_1+E_2)/(m) / 2$ ( for our special case where $\vec{p}\_1+\vec{p}\_2=0$, this
  reduces to $(E_1+E_2)/(2m)$ ). Thus we get $dL/dE_1$ and $dL/dE_2$. Finally,
  since each electron’s energy $E_i$ in our toy equals its raw ADC (and $dL = HitEnergy_i \cdot dE_i$ in digitization), we propagate back to $g$:
  $$\frac{\partial L}{\partial g} = \sum_{i} \frac{dL}{dE_i} \frac{\partial E_i}{\partial g} = \sum_{i} \frac{dL}{dE_i} HitEnergy_i.$$
  In code we
  sum over the two electrons. This shows conceptually how changing the gain
  would change the final likelihood. In a full system, one could use automatic
  differentiation or adjoint methods to do exactly this for many calibration
  parameters or detector settings. For instance, one could imagine applying
  operator-overloading AD to the digitizer or even GEANT4 itself so that $dL/dg$
  (and $dL/d\theta$) come out automatically.

## Beyond Histograms

Traditional analyses summarize data in histograms and use binned likelihoods.
However, histograms are fundamentally non-differentiable: a tiny change in a
measured value can jump it between bins, so gradients vanish or become
ill-defined. Differentiable inference therefore requires smooth, continuous
representations of distributions. Options include:

* Unbinned likelihoods: use the exact analytic form of probability densities on
  continuous data.
* Kernel density estimates: replace each event by a smooth Gaussian (or other
  kernel) contribution to histograms, effectively "smearing" bin edges.
* Parametric models: e.g. fit observables to an analytical function or
  normalizing flow.
* Neural density estimators: train a model (normalizing flow, autoencoder, etc.)
  to approximate the distribution in a differentiable way.

For example, instead of hard binning $m_{ee}$, one could use a binned kernel
density estimate (bKDE), where each event deposits a smooth Gaussian of width
$\sigma_{\rm KDE}$ into the histogram. The kernel width is itself a
differentiable parameter that controls smoothness. Modern tools like pyhf
already allow building histogram-based models using libraries like PyTorch or
TensorFlow , which means the histogram counts and likelihood can be placed
inside an autodiff graph. In summary: we avoid the pitfall of non-differentiable
bins by never treating a histogram as a final, immutable object. Instead, we
keep the distribution continuous or explicitly differentiable.

## A New Analysis Paradigm

This differentiable approach changes the paradigm of analysis design. In a
conventional pipeline, one might ask "**do these binned histograms agree with
the hypothesis?**" and manually tweak cuts or calibrations by hand. In a
differentiable pipeline, the question is instead "**how much information about
the physics parameters survives each step, and how can we maximize it?**"
Gradients and scores become first-class: we can optimize selection cuts,
calibration constants, and even detector design parameters using gradient-based
algorithms. For instance:

* Optimizing selections: A cut (e.g. $E>10$ GeV) is non-differentiable as a step
  function, but we can replace it with a smooth sigmoid and then optimize its
  threshold via gradients.
* Choosing observables: Rather than fixed histograms, one could use automated
  techniques (e.g. classifier scores or neural summarizers) whose output is
  differentiable.
* Tuning calibrations: As our example shows, we can update a gain $g$ by
  gradient ascent on the overall likelihood. In practice, this could refine
  alignment, energy scales, or timing calibrations directly for analysis
  performance, rather than purely from calibration data.
* Detector R&D: Even detector geometry or material choices could in principle be
  treated as parameters to optimize (an idea borrowed from inverse graphics).

This is similar to the mindset in differentiable rendering: in graphics, one
asks "**what scene parameters best produce the observed image?**" by
backpropagating through the renderer. In HEP, we ask "**what theory and detector
parameters best produce the observed data?**" and compute $\partial
L/\partial\theta$, $\partial L/\partial g$ to find out. The key shift is that
the entire pipeline is coupled: we optimize end-to-end for the final physics
objective, rather than optimizing each stage in isolation.


## Differentiable Rendering Analogy

There is a close analogy to computer graphics. In ray tracing, one starts with
scene parameters (object shapes, textures, lights) and computes pixel
intensities through a rendering pipeline (ray intersections, shading, camera
sensor). Differentiable renderers compute
$\partial(pixels)/\partial(\text{scene parameters})$, enabling inverse
design (e.g. fitting a 3D model to an image). Our HEP pipeline is like a "ray
tracer" for physics: physics parameters $\to$ particle transport (showering)
$\to$ detector readout (digitization) $\to$ observables (hits, energies, tracks)
$\to$ statistical inference. A differentiable HEP pipeline thus computes
derivatives of the final likelihood or loss w.r.t. upstream
parameters. Techniques developed in differentiable rendering -- such as
variance-reduced gradients for Monte Carlo, surrogate submodels for expensive
paths, or hybrid exact/smooth approximations -- are directly transferable.


## Goals and Best Practices

This minimal code serves as a Rosetta stone, linking traditional HEP software
concepts to differentiable inference ideas. Our aims are:

* Transparent end-to-end differentiability: Demonstrate in simple code how the
  entire pipeline can be seen as one differentiable function.
* Backward sensitivity propagation: Explicitly show how sensitivity (gradients)
  flows backward from the likelihood to calibration parameters.
* Modular structure: Keep each toy function aligned with a real subsystem so it
  can be replaced by an actual implementation later (e.g. swapping in GEANT4 for
  `simulate_shower`).
* Responsible surrogates: If we use any surrogate or approximation (like our toy
  shower), it should be physics-informed and validated. We emphasize that
  surrogates are tools, not replacements. For instance, our `simulate_shower`
  obeys basic shower physics and is only used as a stand-in; any deployment
  would require checking it against full simulation.
* Validation & uncertainty: Future work should incorporate uncertainty estimates
  (e.g. outputting not just a mean but a probability distribution) and validate
  surrogate components in regimes where they are applied.

**This is not a call to blindly replace GEANT4 with a neural net; rather, it is
a blueprint for hybrid, information-optimal pipelines. One could combine exact
physics where needed, and inject differentiable approximations only where they
are accurate and speed up optimization.** Instead of replacing or
differentiating GEANT, we propose to extract and parameterize the
low-dimensional sensitivity operators that GEANT induces on physics observables,
enabling end-to-end gradients at CMS scale without sacrificing fidelity.  We do
not differentiate GEANT; we run it a controlled number of times to extract
low-dimensional response derivatives that serve as differentiable detector
operators.


## Learning from Past Surrogates

Previous efforts to use surrogates in HEP sometimes stumbled because the
surrogate lacked fidelity or ignored uncertainties. Our approach explicitly
avoids those pitfalls:

* Physics-informed models: Our toy components are built with physics knowledge
  (e.g. energy deposition laws) rather than arbitrary fits. This is in line with
  modern trends to use physics-aware networks.
* Integrated workflow: We embed the differentiable components inside the full
  analysis chain, rather than as standalone black boxes. The entire chain (from
  detector to likelihood) is differentiable, so intermediate validation is
  always possible.
* Probabilistic outputs: Though our toy uses a point likelihood, a real
  differentiable pipeline would output full probability densities or scores. For
  example, instead of a single energy prediction, a surrogate calorimeter could
  output a distribution of possible showers.
* Validation and uncertainties: Any surrogate in the pipeline must be validated
  against full simulation in the regions of interest, and its uncertainty must
  be tracked. This separates "tool" from "trust": the surrogate accelerates the
  pipeline, but we know when and where its approximations hold.
* Modularity & interpretability: By preserving the traditional module structure,
  each stage remains interpretable. We combine the strengths of modular analysis
  with end-to-end optimization.

**In short, we design surrogates to enable differentiation and optimization, but
not to obfuscate physics. We always keep a clear path back to the true physics
model.**


## Conclusion

This minimal pipeline code is intended as a teaching example and a starting
point for future development. It shows that with a few simple, differentiable
building blocks we can link theory, simulation, reconstruction, and analysis
into one coherent graph. By doing so, we unlock gradient-based tools across the
board. As recent studies have demonstrated, end-to-end differentiability can
greatly enhance analysis sensitivity. Our hope is that this example serves as a
"Rosetta stone", making the path from toy modules to real HEP software
explicit. Each function here is a signpost to a real subsystem; replacing them
one by one with actual implementations (autodiffed GEANT4, full recon, advanced
likelihoods) would yield a practical differentiable analysis chain. The lesson
is that differentiable inference is not magic: it’s about preserving information
and optimizing every part of the pipeline for the final physics goal.
