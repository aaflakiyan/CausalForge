# Universal Teacher Distillation: architecture, evidence and validation

**Status: implemented research baseline, not a universal pretrained VLA.** The CPU student, provenance-preserving evidence format, curriculum and grounded reasoning gate have runnable tests. The MuJoCo benchmark trains on two *scripted* dual-UR5e teachers and executes its trained student against held-out goals. Actual pretrained teacher checkpoints, cross-embodiment transfer and Cosmos 3 inference are not asserted by these tests.

## Why merging all VLA weights would be unsound

OpenPI/π₀, GR00T, LeRobot/SmolVLA, Octo and OpenVLA differ in backbones, action heads, tokenisation, robot sensors and control conventions. Concatenating weights or treating arbitrary output vectors as identical commands is invalid. Even perfect action conversion cannot guarantee that one finite model preserves every skill learned by multiple larger specialists.

CausalForge therefore separates:

```text
             OpenPI     GR00T     LeRobot     other
                \         |          |       /
              Versioned native prediction records
                   |
             Checked physical adapters
               [UR5e] [humanoid] [...]
                   |
          Measured outcome + independent safety checks
                   |
          Physical hypothesis and counterfactual probes
                (e.g. Cosmos3 Reasoner)
                   |
         Verified teacher selection for TRAINING
                   |
          Shared semantic task-feature projection
                   |
        +-----------+----------------+
        |                            |
   UR5e action head              Humanoid action head
        |                            |
     MuJoCo test                  Embodiment-specific test
```

**The original source action stream is retained in full**, along with teacher identity, architecture, checkpoint identifier, observation digest, dataset split, action specification and measured outcome. It is never silently converted into an incompatible common motor vector.

## Actual implemented interfaces

| Component | What it does | Limits |
|---|---|---|
| `TeacherEvidence` | Versioned JSONL, native output + checked action, provenance and measured outcome | A future dataset ingest step must supply honestly measured outcomes |
| `ExplicitActionMapping` | Exact robot/joint/frame/unit mapping with opt-in rate conversion | Does not solve cross-robot inverse kinematics |
| `UnifiedStudent` | Shared PCA task-feature projection and separate ridge action heads | A CPU baseline, not a transformer VLA |
| `disagreement_curriculum` | Selects high-disagreement *train* episodes for future physical probes | Disagreement does not establish which teacher is correct |
| `ReasoningGate` | Requires grounded fact citations; can suggest physics experiments | Text does not issue control commands or automatically become labels |
| `Cosmos3ReasonerClient` | Optional OpenAI-compatible text reasoning API client | Needs externally served authorised Cosmos3 model; CI uses mocked transport |
| `ur5e_teacher_student` | Real MuJoCo scripted-teacher outcomes → student fitting → held-out UR5e execution | Not real pretrained VLA training |

The CPU baseline uses a shared projection fitted only from the training split, plus a per-embodiment action head. Its loss is ordinary supervised action reconstruction from the best **measured-safe** teacher in each training episode. Test outcomes are used only for **offline scoring**, never online routing. It will not preserve teacher skills that are absent from the training coverage.

## Try it on CPU: no GPU, MuJoCo or model downloads

```bash
git clone https://github.com/aaflakiyan/CausalForge.git
cd CausalForge
python -m pip install -e .
causalforge distill-demo --episodes 80 --out runs/universal-student
```

The command produces:

```text
runs/universal-student/
  synthetic_teacher_evidence.jsonl   # source predictions retained
  student.json                      # shared encoder + two action heads
  report.json                       # train/test, provenance hash, MSE, curriculum
```

This first demo explicitly uses **synthetic generators**, two deliberately different action dimensions, and measured-like **synthetic** labels. It is not GR00T/π₀ inference or measured robot success.

## Actual MuJoCo training and held-out evaluation

```bash
python -m pip install -e '.[mujoco]'
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie fetch --depth 1 origin 98a70c569c4c1b6da467a20012f766ee1f8dc521
git -C third_party/mujoco_menagerie checkout FETCH_HEAD
git -C third_party/mujoco_menagerie sparse-checkout set universal_robots_ur5e

python -m vla_council.causalforge.ur5e_teacher_student \
  --menagerie third_party/mujoco_menagerie \
  --train 10 --test 4 \
  --out runs/real-ur5e-distillation
```

This samples nearby, independently specified 3D dual-arm targets, executes two **scripted** joint-target teachers in MuJoCo, measures each one's actual reaching error and contacts, selects training labels from observed-safe candidates, trains the student, then evaluates the student's own joint targets under fresh MuJoCo physics on four unseen goals. Real results and student weights are archived as CI artefacts. No test outcome is available to the training label selector.

## Cosmos 3 and structured physical reasoning

NVIDIA's [Cosmos 3 platform](https://github.com/NVIDIA/cosmos) now provides unified reasoning and multimodal generation/action capabilities. It is a promising *external* component for proposing physical hypotheses, expected contact outcomes and diagnostic experiments, not an authority that can bypass robot safety.

The optional `Cosmos3ReasonerClient` accepts an OpenAI-compatible Reasoner chat endpoint, consistent with [NVIDIA's Cosmos 3 Reasoner serving recipes](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/reasoner/README.md). It sends **only supplied structured facts and task/proposal names**; it expects strict JSON containing:

```json
{
  "hypothesis": "unexpected slip may be related to contact friction",
  "verdict": "uncertain",
  "evidence_refs": ["measured_slip_m", "measured_normal_force_N"],
  "suggested_probe": "repeat at reduced acceleration under controlled friction",
  "rationale": "observed slip alone does not identify its physical cause"
}
```

The gate rejects nonexistent, duplicate or future evidence references. A reasoner cannot submit executable joint commands through this interface.

```python
import causalforge

reasoner = causalforge.Cosmos3ReasonerClient(
    base_url="http://localhost:8000",
    model="YOUR_DEPLOYED_COSMOS3_REASONER_MODEL",
)
gate = causalforge.ReasoningGate(reasoner)
answer = gate.assess(
    task="diagnose slip after manipulation",
    facts=[
        causalforge.PhysicalFact("measured_slip_m", .02, "m", "mujoco_measurement"),
        causalforge.PhysicalFact("measured_normal_force_N", 12., "N", "sensor"),
    ],
    proposals=["teacher_pi", "teacher_groot"],
)
```

Running the snippet requires a separately configured Cosmos3 endpoint and correct model name. CPU CI replaces the network response with a controlled test double. Do **not** send proprietary data or credentials to unknown external endpoints.

### Where physical reasoning should affect learning

1. Record a real failure and facts (joint state, slip, force, collision, delay).
2. Ask one or more reasoners for **testable competing hypotheses**, with evidence references.
3. Rank candidate diagnostic experiments by disagreement and cost.
4. Execute admissible probes in MuJoCo (or a protected hardware setup).
5. Accept only **measured** outcomes into the training set; reject unsupported explanations.
6. Train the student with mixed embodiments, teacher reliability and task context.
7. Re-evaluate on held-out tasks, disturbances and robot embodiments.

The present code implements the fact gate and offline disagreement scheduler, and the UR5e demonstration uses measured training outcomes. **Automatic reasoning → experimental design → physics rollout → online retraining is a next research milestone**, not a current feature.

## Which external components should be integrated next?

| Component | What it would contribute | Priority |
|---|---|---|
| [GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) | Humanoid and bimanual VLA teacher; embodiment-aware fine-tuning | High |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | Flow-matching continuous action teacher | High |
| [LeRobot/SmolVLA](https://github.com/huggingface/lerobot) | Lightweight training, demonstrations and common dataset format | High |
| [Cosmos 3](https://github.com/NVIDIA/cosmos) Reasoner | Explicit physics hypotheses and experiment proposals | High |
| Cosmos 3 Generator / world-action models | Action-conditioned counterfactual rollouts (require physics verification) | Medium |
| [CrossFormer](https://proceedings.mlr.press/v270/doshi25a.html), [UniVLA](https://arxiv.org/abs/2505.06111) | Task-centric latent representations and stronger cross-embodiment model design | Research |
| MuJoCo / Menagerie | Deterministic physical ground truth, perturbations and safety metrics | Essential |

Some checkpoint and code licences require separate review before redistribution or commercial use. In particular, do not assume Cosmos and other model weights inherit the CausalForge repository's MIT licence.

## What success would actually mean

A research-grade claim needs all of the following: trained and version-pinned checkpoints, comparable robot observations/actions, genuine held-out tasks, independent task-success measurements, physics and intervention costs, negative-control and no-reasoning baselines, per-teacher skill retention, out-of-distribution failures, and uncertainty estimates over repeated seeds.

The system can **retain access to teachers as fallbacks** when a compact student is uncertain. That preserves access to capabilities at a compute and latency cost. It does *not* turn all specialist knowledge into guaranteed performance from one small neural network.
