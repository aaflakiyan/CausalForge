# Validation evidence — Lab 01

## What has been verified

The Academy GitHub Actions workflow installed the **real MuJoCo 4.0.1 CPU bindings**, ran unit + API + physics integration tests and completed a four-target benchmark on Ubuntu, Windows and macOS, with CPython 3.11 and 3.13.

- Original tested commit: `1dff6cdf3dd2d3d4c4f5fd3bde5e3faa70091750`.
- GitHub Actions run: https://github.com/aaflakiyan/CausalForge/actions/runs/43980128288
- Matrix: 6/6 jobs successful (Ubuntu, Windows, macOS × Python 3.11, 3.13).
- Benchmark: 4/4 target reaches succeeded on an Ubuntu Python 3.13 job (threshold = 0.025 m; 1200 steps, 0.005 s/step).
- Recorded final errors for the four targets: 0.016296 m, 0.019453 m, 0.008721 m, 0.015253 m.

These observations apply to the *simplified two-joint reaching model* under the benchmark conditions. They do **not** demonstrate object grasping, camera-based perception, safe operation on hardware, domain transfer, or robust manipulation.

## Reproduce

```bash
cd academy
python -m pip install -e '.[dev]'
python -m pytest -q
physical-ai-lab --steps 1200 --out results/benchmark.json
```

The generated JSON records the chosen targets, final errors and success fraction. The browser UI additionally visualises error history; a full browser interaction test across multiple browsers is not yet part of CI.

## Release gates for Lab 02

1. All numerical and physical simulation tests pass on supported environments.
2. Label simulated versus real sensor observations clearly.
3. Publish evaluation settings, seeds, units and known failure cases.
4. Verify UI controls on a representative browser before public deployment.
5. Make any hardware command interface explicitly opt-in and safety reviewed.
