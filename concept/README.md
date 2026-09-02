# Minimal Differentiable HEP Pipeline — Teaching README

## Introduction

This repository contains a **minimal, teaching-oriented C++ implementation of an
end-to-end high-energy physics (HEP) analysis pipeline**. The goal is not
realism or performance, but conceptual clarity. To show how a modern HEP
analysis can be viewed as a *single differentiable computational graph*
connecting theory, detector simulation, reconstruction, and statistical
inference. The code demonstrates:
* a forward pipeline from particles to reconstructed observables;
* an explicit likelihood at the analysis level;
* and a manual backward pass that propagates sensitivity (gradients) back to
  detector calibration parameters.

Each routine is intentionally small and maps cleanly onto a **real production
subsystem** used in experiments such as CMS or ATLAS. The toy implementation
acts as an index or placeholder for progressively replacing each step with a
realistic one.

---

## Basic Mathematical Picture (Very Simple)

At its core, an experiment implements a mapping (parameters) → (data). More
explicitly:

$\theta$ (physics parameters) → simulation → detector response → digitization (with calibration parameters g) → reconstruction → observables x → likelihood L(x | θ, g)


The likelihood tells us how compatible the observed data are with a hypothesis.
Inference traditionally proceeds by scanning parameters and comparing likelihood
values.

In a **differentiable pipeline**, we additionally compute how the likelihood
*changes* when parameters change:

$$\frac{\partial L}{\partial g}, \qquad \frac{\partial L}{\partial \theta}$$

Using the chain rule, these derivatives can be propagated backward through the
entire pipeline. This allows direct, gradient-based optimization of calibration
parameters, selections, or even detector design choices, rather than relying on
brute-force scans or heuristic tuning.

---

## Mapping Toy Routines to Real Experimental Subsystems

Each function in the code corresponds directly to a well-defined subsystem in a
production experiment.

### `simulate_shower(...)`
**Role:** Generates electromagnetic calorimeter hits from a truth-level electron
using a parametric model.

**Production equivalent:**  
- Event generation: **MadGraph5_aMC@NLO**, **Pythia**
- Detector simulation: **Geant4** or **HepEmShow**
- Data products: `HepMC::GenEvent`, `G4Step`, `PCaloHit`

This stage models particle transport and energy deposition in detector material.

---

### `digitize_hits(...)`
**Role:** Maps energy deposits to calorimeter cells, applies a gain factor and
electronic noise, and produces digitized signals.

**Production equivalent:**  
- **Digitization / Electronics Simulation**
  - CMS: `EcalElectronicsSim`, `EcalDigiProducer`
  - ATLAS: LAr digitization
- Uses calibration constants (gains, pedestals) stored in conditions databases.

This is where detector calibration parameters naturally live.

---

### `reconstruct(...)`
**Role:** Builds electron candidates directly from digis, preserving links to
detector cells.

**Production equivalent:**  
- **Reconstruction**
  - CMS: `reco::SuperCluster`, `reco::GsfElectron`, Particle Flow
  - ATLAS: topoclusters, electron reconstruction
- Responsible for clustering, identification, and energy/momentum estimation.

---

### `likelihood(...)` and `invariant_mass(...)`
**Role:** Computes a reconstructed invariant mass and evaluates a simple
Gaussian likelihood.

**Production equivalent:**  
- **Physics analysis and inference**
  - RooFit / RooStats
  - HistFactory, pyhf
- Constructs statistical models and performs fits or hypothesis tests.

---

### `backward_gain(...)`
**Role:** Manually applies the chain rule to compute the derivative of the
likelihood with respect to the digitizer gain.

**Production equivalent:**  
- **Differentiable calibration / adjoint methods**
- Potential implementations:
  - Adjoint or autodiff-enabled detector components
  - Differentiable surrogates validated against full simulation
  - Matrix-element reweighting and score-based estimators

This is the conceptual heart of the differentiable pipeline.

---

## Replacing Histograms as the Surrogate

Traditional histograms act as an implicit surrogate for the underlying
probability distribution: continuous observables are discretized into bins, and
bin counts are treated as sufficient statistics. While effective for
visualization and many classical analyses, histograms are fundamentally
ill-suited for differentiable pipelines. Binning discards information,
introduces non-differentiable boundaries, and forces arbitrary design choices
(bin widths, edges) that directly limit sensitivity. In a differentiable
context, histograms cannot serve as the primary interface between simulation and
inference. They must be replaced by representations that are continuous and
differentiable, such as unbinned likelihoods, smooth kernel density estimates,
parametric distribution models, or modern conditional density estimators. These
alternatives preserve more information and allow gradients to flow cleanly from
the analysis back to upstream parameters.

---

## How This Transforms the Analysis Paradigm

Removing histograms as the central abstraction fundamentally changes how
analyses are designed. Instead of treating the analysis as a terminal step that
consumes fixed distributions, the analysis becomes *coupled* to the entire
pipeline. The question shifts from “how well do these binned shapes agree?” to
“how much information about the parameters of interest survives each
transformation?” Observables are chosen and optimized based on information
content rather than tradition or visual clarity. Gradients, scores, and
likelihoods become first-class objects, enabling automatic optimization of
calibrations, selections, and even detector configurations. This represents a
deep but simple change: the analysis evolves from a static comparison exercise
into a dynamic, end-to-end inference problem where every component can be tuned
to maximize sensitivity.

---

## Parallel with Ray Tracing and Differentiable Rendering

There is a direct analogy between HEP pipelines and computer graphics rendering:

- **Ray tracing:** scene parameters → light transport → pixels  
- **HEP:** physics parameters → particle transport → detector signals → observables

Differentiable ray tracers compute derivatives of pixel intensities with respect
to scene parameters, enabling inverse rendering and optimization. Similarly, a
differentiable HEP pipeline computes derivatives of likelihoods with respect to
physics, detector, and calibration parameters. Both domains involve stochastic
Monte Carlo processes, expensive forward simulations, and the need for adjoint
or surrogate-based differentiation. Many techniques developed for differentiable
rendering—variance reduction, pathwise derivatives, hybrid exact/surrogate
models—transfer directly to HEP.

---

## What We Aim to Achieve

- Demonstrate end-to-end differentiability using a transparent, minimal
  codebase.
- Show how sensitivity propagates backward through the pipeline to calibration
  parameters.
- Provide a modular structure where each toy component can be replaced by a real
  production system.
- Establish a framework where surrogate models are used responsibly, with
  validation and uncertainty awareness.

This is not a proposal to replace full simulation wholesale, but a blueprint for
**hybrid, information-optimal pipelines** that combine exact physics, validated
surrogates, and differentiable inference.

---

## Why This Is Not a Repeat of Past Surrogate Failures

Earlier surrogate attempts often failed due to lack of fidelity, poor
uncertainty modeling, and weak integration with downstream analyses. The modern
approach differs by:
- using physics-aware architectures that respect conservation laws,
- producing probabilistic outputs rather than point estimates,
- validating surrogates against full simulation in targeted regimes,
- and embedding them inside the full inference chain rather than treating them
  as standalone replacements.

Surrogates are tools, not substitutes for physics. When used carefully, they
enable differentiation, optimization, and sensitivity gains that were previously
inaccessible.

---

## Final Remark

This repository is intended as a **conceptual Rosetta stone**: a
smallest-possible working example that connects the language of traditional HEP
software with modern differentiable inference ideas. Each function is a signpost
pointing toward a real subsystem, making the path from toy model to production
system explicit and teachable.
