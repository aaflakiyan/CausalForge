# Installation: Windows, Linux, macOS, Codespaces

## Requirements

Python 3.11–3.13, internet for the first package installation, and an ordinary computer. No NVIDIA GPU, ROS 2 or Isaac Sim needed. Tested automatically on Windows, macOS and Ubuntu in GitHub Actions once the pull request workflow runs. A green CI badge should only be claimed after the workflow has passed.

From the **repository root**:

```bash
cd academy
python -m venv .venv
```

Activate the virtual environment:

- **Windows PowerShell:** `.venv\\Scripts\\Activate.ps1`
- **Linux / macOS:** `source .venv/bin/activate`

Then install and start:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
python -m uvicorn physical_ai_lab.app:app --port 8000
```

Open **http://localhost:8000** in your browser. In Codespaces, forward port 8000, open the forwarded URL and ensure visibility is private unless you intentionally want to share the single-user simulation.

**Windows note:** use `py -3.11 -m venv .venv` if `python` points to an unsupported installation. PowerShell execution policy can prevent activation; you may instead run `.venv\\Scripts\\python.exe -m pip ...` and `.venv\\Scripts\\python.exe -m uvicorn ...` without activation.

**macOS note:** use a supported 3.11+ Python distribution appropriate to Apple Silicon or Intel.

## Zero-install viewing (inspection only)

[MuJoCo Live](https://live.mujoco.org/) can open MJCF models in a browser. Load `academy/models/planar_arm.xml` by dragging the file into MuJoCo Live. This lets you view the model without Python, but **does not run the Academy's custom controller or sliders**. For the full interactive lab use the Python server, including via Codespaces.

## Troubleshooting

- **ModuleNotFoundError:** ensure your venv is activated, then reinstall `-e '.[dev]'` from `academy`.
- **Port in use:** change `--port 8001` and browse to the same port.
- **HTTP 503 / engine unavailable:** the server started without a functional MuJoCo installation. Run `python -c "import mujoco; print(mujoco.__version__)"`.
- **Simulation gets stuck:** press Reset and lower controller gain or sensor noise. Capture `results/benchmark.json` before filing an issue.
- **Codespaces idle or closed:** Codespaces is not a permanently hosted public site; reopen it and start the server.

For course publication, do not expose the single-user API to the public internet without authentication and session isolation.

## Devcontainer integration

The repository has `.devcontainer/physical-ai-academy/devcontainer.json`. On GitHub, open **Code → Codespaces → ... → New with options**, select the Academy configuration when available, and create the environment. The browser interface uses forwarded port 8000. If the setup UI does not expose named configurations, use the manual Codespaces instructions above.
