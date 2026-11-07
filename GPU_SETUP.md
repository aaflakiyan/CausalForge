# Running CausalForge with MuJoCo and GPU hardware

**MuJoCo physics alone does not require a GPU.** CausalForge's current CI runs its toy pushing benchmark on GitHub's standard CPU runners. A CUDA GPU is useful for pretrained VLA inference and training; no pretrained GR00T/pi0 results are claimed here.

## Verify existing MuJoCo experiments

Open the repository's **Actions → Tests** workflow. The `mujoco` job installs `pip install -e '.[mujoco]'`, runs unit tests, executes physical-distillation experiments and uploads a `mujoco-results` artifact. The numerical outputs are small-scale physics checks, not humanoid VLA results.

## Use your own Linux CUDA machine or rented GPU

A Linux machine with a compatible NVIDIA driver and PyTorch CUDA build is the most straightforward setup. A cloud GPU instance from a provider of your choice also works. Consider starting with a 24 GB GPU for smaller policy smoke tests; larger foundation-model checkpoints may require 48–80 GB or multiple GPUs depending on the model, precision, context and batch size. Actual memory needs must be measured, not assumed.

On that machine, in its own terminal (not in ChatGPT):

```bash
git clone https://github.com/aaflakiyan/CausalForge.git
cd CausalForge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[mujoco]'
# Install the correct CUDA-enabled PyTorch wheel from pytorch.org for your platform.
python -m vla_council.causalforge.gpu_preflight --require-gpu
python -m unittest discover -s tests -v
python -m vla_council.causalforge.physical_distillation --backend mujoco --train 36 --test 24 --out runs/physics-on-gpu-host.json
```

The existing `physical_distillation` benchmark is a toy synthetic-teacher experiment; running it on a GPU machine **does not** use the GPU for VLA inference. Before comparing actual teachers we must integrate supported checkpoints into one identical robot embodiment and common observation/action format, then run held-out tasks with matched training and probing budgets.

## How to share compute safely

Never paste a password, SSH private key, GitHub personal access token, cloud API key, or Hugging Face token into chat. Use the private GitHub repository as the exchange point for code and non-sensitive result files. You can run a GPU machine yourself, or use ChatGPT Work's supported computer and GitHub connection if your account offers the relevant workflow. This conversation's GitHub connector can edit files and inspect CI results; it does not provide an SSH session into your computer or a rented GPU.

For large results, upload artifacts to GitHub Actions or an appropriate private storage location, taking care not to publish patient data, model credentials or proprietary datasets. Keep training outputs and model weights out of ordinary git history.

## Next acceptance gate

1. Obtain **two legitimately accessible checkpoints** licensed for the planned experiment.
2. Demonstrate both can produce actions for the **same embodiment** with the same frame, action units, horizon and controller.
3. Record model versions, transforms, inference precision, GPU memory, latency and every action prediction.
4. Evaluate standalone teachers, their council, ordinary distillation, and causal active-selection distillation under paired seeds and matched costs.
5. Report uncertainty and failure modes, including out-of-distribution physics and the possibility that CausalForge does not improve over baselines.
