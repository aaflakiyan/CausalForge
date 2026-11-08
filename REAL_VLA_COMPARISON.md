# Connecting genuine VLA checkpoints and fairly comparing tasks

**Two separate, honest evaluation tracks.** The CausalForge repo already runs a real **dual-UR5e MuJoCo** simulation, but its checked-in GR00T, OpenPI and LeRobot demos use **mock provider outputs** generated from an IK solution. They validate the interfaces, NOT pre-trained model abilities.

This addition attempts an authentic checkpoint comparison in a **native task**:

| Model | Native evaluation task | Model category |
| --- | --- | --- |
| [n3puiol/smolvla_pusht](https://huggingface.co/n3puiol/smolvla_pusht) | PushT | Real finetuned vision-language-action checkpoint |
| [aadarshram/act_pusht](https://huggingface.co/aadarshram/act_pusht) | PushT | Real trained action-chunking imitation policy; **not a VLA** |
| [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht) | PushT (optional) | Real trained diffusion imitation policy; **not a VLA** |

The native PushT simulator uses **Pymunk**, **not MuJoCo**. It has two-dimensional end-effector actions. **Do not route its checkpoint actions into a six-joint UR5e**; the robot, control mode and semantics differ. Treat this as a real-checkpoint inference track complementary to MuJoCo mechanical validation.

## Run real pretrained models on their native task

A version-matched LeRobot installation and network access to Hugging Face model files are required. The CI runner pins `lerobot==0.4.4` to reduce upstream API drift. On a suitable Python 3.11 environment:

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
