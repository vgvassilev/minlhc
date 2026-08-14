# MODE6 Workshop Abstract

**Sixth MODE Workshop on Differentiable Programming for Experiment Design**
Kolymbari, Crete, 1 to 7 September 2026. <https://indico.cern.ch/event/1655754>
Track: Methods and Tools · Format: contributed talk

---

## Title

**Trust, then scale: checkable gradients in a minimal end-to-end pipeline**

## Abstract

End-to-end differentiability is changing how we think about analysis and
experiment design: instead of scanning parameters by brute force and tuning
calibrations by hand, we can let gradients of the likelihood flow back through
reconstruction, digitization, and detector response, and optimize calibrations,
selections, and eventually design for the physics objective directly. However,
at the scale of a full simulation-and-reconstruction stack it is hard to tell
whether a given differentiable method genuinely works or only appears to: it
requires a lot of work upfront, ground truth is unavailable, brute-force
baselines are unaffordable, and the failure modes hide.

Our prototype deliberately reduces complexity. We have built a minimal
end-to-end pipeline (generation, showering, digitization, reconstruction,
likelihood) small enough that we can check every gradient by hand, yet arranged
so each stage maps one-to-one onto a real subsystem, so the integration path
survives. Part of the motivation is pedagogical: the coupling between
calibration and physics inference is far easier to see when the whole chain fits
on a screen. The stronger reason is research discipline: a small system is where
we can obtain ground truth, afford the brute-force baseline, and actually prove
that a method is right rather than merely plausible.

The talk is organized around what this testbed establishes, and what it does
not. Where the downstream is analytic, the adjoint matches finite differences to
machine precision, a coding check, not a discovery, but a necessary one. When a
parameter lives inside a stochastic simulator we treat as a black box, we
extract a frozen local response operator from fixed-seed runs; two independent
extractions agree to well under a percent, and (the honest part) we can
*measure* the narrow window in which that operator still tracks a brute-force
rerun, rather than assume it does. Within that window a joint fit recovers a
parameter buried inside the simulator alongside a calibration gain, to a few
percent, where the usual fix-the-nuisance-and-scan cannot reach the buried
parameter at all. It does so without re-simulating at every step, taking its
gradients from cached events and the frozen operator.

We are explicit about scope: this is a closed-world injection-recovery study with
detector noise switched off, the operator is only locally valid and must be
refreshed, and the efficiency is structural rather than a measured speedup on
anything real. What we think it earns is a small, checkable object with which to
reason about end-to-end optimization before paying the cost of scale, and a
shared substrate for exactly the cross-domain conversation MODE convenes. We
would welcome feedback from detector, reconstruction, and
differentiable-programming colleagues on which of these results survive contact
with their systems, and which break first.

---

## Alternative titles

1. Keeping a differentiable analysis small enough to prove
2. Trust, then scale: checkable gradients in a minimal end-to-end pipeline
3. End-to-end differentiability, tested where the truth is known
4. Small enough to prove, structured enough to grow
5. Frozen response operators in a differentiable analysis, and where they stop holding
6. Provable before scalable: a minimal differentiable analysis pipeline

---

## Provable basis (from `concept/opaque_oracle_demo.cpp`)

Every quantitative claim in the abstract is bounded by a passing diagnostic in
the demo. Verified by building and running it.

| Claim in abstract | Evidence | Honest caveat |
| :--- | :--- | :--- |
| Adjoint = finite differences to machine precision | T1a: rel. error 9.9e-13 | Only on the *analytic* downstream (gain path); a coding check, not a result |
| Extracted operator is stable | T2: two independent extractions agree to 0.36% | Shows low variance, not correctness |
| Measured validity window | T3: valid for μ∈[0.96, 1.08] at 5% agreement | Window is narrow → operator must be refreshed during optimization |
| Adjoint through operator tracks brute force | T1b: rel. error 0.20 (< 0.25 tol) | ~20% agreement, statistical; shrinks with more seeds; NOT machine precision |
| Joint fit recovers buried parameter + gain | T4: μ 0.82 vs 0.85 (~4%), g 0.94 vs 1.0 (~6%) | Closed-world injection-recovery; noise off; MLE offset within stochastic floor |
| Fix-and-scan cannot reach the buried parameter | T5 + classical scan holds μ fixed at nominal | The g-bias itself is marginal (~1%); the real gap is the un-estimable nuisance |
| Data efficiency | `scan_mu` (re-sims oracle per point) vs `linearised_logL` (cached events + frozen operator) | Structural only; J extraction costs ~1000 oracle runs and needs refresh; no wall-clock speedup measured |

**Out of scope / not claimed:** detector noise (off throughout), model
misspecification (targets generated at truth with the same model), any real
detector, any benchmarked speedup, and any use of automatic-differentiation
tooling (all adjoints here are hand-derived).
