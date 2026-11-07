# Implementation status — v0.4

**Implemented and exercised:** synthetic multi-teacher navigation with critics; robot/action compatibility data contracts; independent action-veto council; MuJoCo single-axis model source; Bayesian update and information-gain probe selection; diagnostic benchmark (algebraic and optional MuJoCo backends); a fully connected **synthetic** experiment -> posterior -> synthetic teacher action -> supervised ridge student -> held-out evaluation pipeline; repeated-seed comparisons; automated unit and contract tests.

**Validated here:** NumPy/Python unit tests and synthetic experiment/suite commands. See [VALIDATION.md](VALIDATION.md) and [RESULTS.md](RESULTS.md) for exact commands and measured outputs.

**Not validated here:** MuJoCo execution, since its wheel could not be installed in this environment. GitHub Actions includes a separate MuJoCo job that can exercise this code after publishing.

**Not implemented:** inference from real VLA checkpoints, shared Panda environment, large image-based observation datasets, contact-rich manipulation student, a physical humanoid controller, training a full VLA, online safe robot experiments, or system identification beyond the specified two hypotheses. These are project milestones, not completed deliverables.

**Safety:** no hardware use. The council's veto logic is a mockable software guard, not a certified physical safety system.

## v0.5 status

Implemented: paired known/unknown-parameter physics diagnosis benchmark,
heuristic residual rejection and no-op on rejection, equal-duration probe
comparison with force-cost disclosure, a strict callable VLA adapter boundary,
physics regression tests, MuJoCo CI integration and tracked surrogate reports.

**Outstanding blockers:** actual MuJoCo run on a machine with a working
package index; real VLA checkpoints compatible with one embodiment; grasping
and full-body humanoid environments; student training on physically sampled
VLA demonstrations; externally validated safety supervision. None of these
is claimed complete.