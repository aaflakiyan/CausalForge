# Recorded VLA teacher benchmark (experimental)

This is the first bridge from real, externally executed VLA checkpoints to a common **offline action evaluation**. No GR00T, OpenPI or SmolVLA checkpoint is bundled, downloaded or run by this code.

## Contract

Record a JSON Lines file with **one independent episode/state per line** and the following fields:

```json
{
  "episode_id": "trial-001",
  "split": "train",
  "observation": [0.1, 0.2],
  "target_action": [0.3, -0.1],
  "teacher_actions": {
    "groot": [0.28, -0.11],
    "pi0": [0.31, -0.09]
  },
  "action_spec": {
    "robot": "panda",
    "representation": "joint_velocity",
    "ordered_axes": ["joint1", "joint2"],
    "units": ["rad/s", "rad/s"],
    "frame": "robot_base",
    "dt": 0.05
  }
}
```

**The numbers above are illustrative and not actual pretrained-model predictions.**
Supply multiple rows with nonempty `train` and `test` partitions; keep object identities and scenes disjoint between partitions to measure generalisation.

Each teacher must be converted using a *separately validated* robot-specific adapter. The loader refuses differing frame, robot, units, action axes and timestep. Matching shapes alone is insufficient. The original temporal action chunk is **not** retained by this minimal recorder: export a fixed, clearly specified action at a shared decision point. If teacher horizons differ, evaluate alignment independently before claiming comparability.

## Run

```bash
python -m vla_council.causalforge.recorded_vla path/to/teacher_recordings.jsonl --out runs/teachers.json
python -m unittest discover -s tests -v
```

The benchmark reports held-out **action mean-squared error**, including the best individual teacher, teacher average and ridge students trained with different labels. The supervised teacher selector uses ground-truth *training* actions to pick its training labels; it is **not** a learned critic that can select teachers without supervision.

**Do not interpret action MSE as task success, safe execution, robustness, or causal improvement.** The next stage must execute all candidate actions from identically reset MuJoCo states, measure success and contact/collision outcomes, and train using only the designated training split. Run several seeds, matched teacher-query budgets and OOD object conditions. Two pretrained models will also require GPU resources, valid checkpoints and consistent observation/action preprocessing.

## Safety

Only use this code for offline data processing or simulation. Do not pass teacher predictions directly to physical actuators; use an independently validated low-level safety controller and emergency stop.
