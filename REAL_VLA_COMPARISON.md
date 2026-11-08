# Connecting genuine VLA checkpoints and fairly comparing tasks

**Two separate, honest evaluation tracks.** The CausalForge repo already runs a real **dual-UR5e MuJoCo** simulation, but its checked-in GR00T, OpenPI and LeRobot demos use **mock provider outputs** generated from an IK solution. They validate the interfaces, NOT pre-trained model abilities.

This addition attempts an authentic checkpoint comparison in a **native task**:

| Model | Native evaluation task | Model category |
| --- | --- | --- |
| [n3puiol/smolvla_pusht](https://huggingface.co/n3puiol/smolvla_pusht) | PushT | Real finetuned vision-language-action checkpoint |
| [aadarshram/act_pusht](https://huggingface.co/aadarshram/act_pusht) | PushT | Real trained action-chunking imitation policy; **not a VLA** |
| [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht) | PushT (optional) | Real trained diffusion imitation policy; **not a VLA** |

The native PushT simulator uses **Pymunk**, **not MuJoCo**. It has two-dimensional end-effector actions. **Do not route its checkpoint actions into a six-joint UR5e**; the robot, control mode and semantics differ. Treat this as a real-checkpoint inference track complementary to MuJoCo mechanical validation.

## Actual first checkpoint trial: measured outcomes and blockers

The first experimental CI run used **Python 3.11, LeRobot 0.4.4, CPU, PushT seed 42, one episode**. The publicly hosted weights were fetched and passed to LeRobot's native evaluator; no mock actions were injected.

| Model | Checkpoint loaded and evaluated? | Actual PushT metric or blocker |
| --- | --- | --- |
| `aadarshram/act_pusht` | **Yes** | 42.34% maximum shape overlap, **0/1** task success. Native episode video and `eval_info.json` are archived. |
| `n3puiol/smolvla_pusht` | **Yes, with LeRobot 0.3.3** | In an independently reproduced historical runtime: **5.98% maximum target overlap, 0/1 successes**, seed 42; native video and `eval_info.json` saved. LeRobot 0.4.4 still refuses the older checkpoint because `policy_preprocessor.json` is missing. |
| `lerobot/vqbet_pusht` | **No** | Older checkpoint config contains `mlp_hidden_dim`, which the current `VQBeTConfig` parser rejects. **No result** is reported. |

A subsequent historical-runtime test **successfully executed the genuine SmolVLA checkpoint under LeRobot 0.3.3**, including an actual PushT episode and recorded video. The SmolVLA and ACT runs are on the same seeded PushT task, but came from different LeRobot versions, so direct architectural comparisons are still preliminary. Neither achieved a successful episode in the first single-seed trial.

An attempted same-runtime rerun on **LeRobot 0.3.3** successfully reproduced SmolVLA's 5.98% overlap, but **the newer ACT checkpoint then failed to load** because its config contains `pretrained_path`, which LeRobot 0.3.3 does not accept. Consequently a strict same-version comparative result is **not yet available**. We retain ACT's independently verified 0.4.4 rollout and both failure logs.\n\nA real diffusion-policy attempt (`DanqingZ/diffusion_pusht`) was also launched alongside the ACT baseline on native PushT. ACT completed its episode again, but **diffusion inference exceeded the 650-second CPU timeout** and produced no valid native evaluation metrics. It is marked **not evaluated**, not unsuccessful at the task. A GPU or a purpose-built lower-step inference configuration would be needed for a practical comparison.\n\nThe native checkpoint experiment exposed real third-party model-version incompatibilities. These are **not** evidence that SmolVLA or VQ-BeT perform poorly at the underlying manipulation task.

The action-schema preflight also downloaded the genuine checkpoint configuration revisions and found both SmolVLA and ACT PushT checkpoints produce **two-dimensional actions**, so neither may be applied directly to the dual-UR5e 12-joint simulator. This safeguard is automatically tested using live Hugging Face metadata.

The genuine ACT episode result is one trial only, so it is **not** a reliable estimate of performance or evidence about which policy is better.

## Run real pretrained models on their native task

A version-matched LeRobot installation and network access to Hugging Face model files are required. Current upstream checkpoints may need independent version migration; the first SmolVLA and VQ-BeT attempts are documented above. The CI runner pins `lerobot==0.4.4` to reduce upstream API drift. On a suitable Python 3.11 environment:

```bash
pip install -e .
pip install 'lerobot[smolvla,pusht]==0.4.4'

# First run downloads actual weights from Hugging Face
causalforge eval-pretrained --episodes 2 --seed 42 --out runs/pretrained-pusht
```

The equivalent explicit command:

```bash
python -m vla_council.causalforge.native_vla_benchmark \
  --model n3puiol/smolvla_pusht \
  --model aadarshram/act_pusht \
  --episodes 2 --seed 42 --out runs/pretrained-pusht
```

**Outputs:** `comparison.json`, per-checkpoint native `eval_info.json`, evaluator log and any native evaluation videos. If downloading or evaluating a checkpoint fails, the command exits unsuccessfully, records the failure and **never substitutes a mock success rate**.

Run this on a CPU for a small compatibility smoke test. A GPU is strongly preferable for larger evaluations or repeated episodes. A one-episode run cannot establish statistically meaningful model superiority.

GitHub CI: [Real pretrained VLA PushT workflow](.github/workflows/real-vla-pusht.yml) tests the actual checkpoint files and preserves the native outputs. It intentionally distinguishes successful executable checkpoint inference from failed/missing model runs.

## What MuJoCo validates and what it does not

```bash
# Entirely separate reproducible 3-D UR5e physics tasks:
causalforge demo
causalforge demo-vlas
python -m vla_council.causalforge.latency_audit \
  --menagerie /path/to/mujoco_menagerie --out runs/mujoco-delay.json
```

The first, third and student training tests execute genuine UR5e physics. The second is an **interface-only** three-provider mock demonstration.

The existing `VLABackend`, `ExplicitActionMapping`, `SemanticMapper` and `TeacherEvidence` make it possible to add checkpoint predictions to training **only when** a complete matching robot action contract and observation preprocessing are provided. The capability to call `get_action()` or `infer()` is not a claim that pretrained policies trained on DROID, WidowX, SO100 or PushT can control dual UR5e.

## Next: comparison on identical MuJoCo dual-arm tasks

For a defensible direct comparison, collect a common dual-UR5e dataset using the real camera/joint/state/goal streams, train or fine-tune two actual pretrained architectures for the same **12-joint absolute position mode at 50 Hz**, then compare:

1. IK baseline, single trained VLA A and single trained VLA B;
2. a student trained on all their verified demonstrations;
3. a reasoning-assisted council with physically grounded hypotheses, matched intervention budget and no future-information leakage;
4. all of those under held-out goal positions, contact-rich manipulation, delayed commands and collisions.

The **native PushT results cannot be combined with the UR5e reach results** into one winner; their tasks, embodiments, action spaces and success metrics differ.

## GPU/hardware and licensing

As of November 2026, official GR00T N1.7 recommends **16 GB or more GPU VRAM for inference**; training usually needs more. GR00T and OpenPI expose pretrained/fine-tuned checkpoints for specific embodiments, but **not a ready-made dual UR5e 12-joint action checkpoint**. Model weights are subject to their own licences and use constraints. CausalForge does not package them or automatically infer a safe conversion.

A genuine claim of improvement requires independently verified model versions, paired test seeds, similar task setup, baseline comparisons, confidence intervals and many more than two episodes.
