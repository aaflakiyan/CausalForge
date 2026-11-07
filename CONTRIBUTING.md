# Contributing

Contributions are welcome, especially reproducible failure cases, tests, unit/frame adapters, benchmark baselines and MuJoCo scenario validation.

1. Create an issue describing the expected behaviour, physical assumptions and reproducible test.
2. Add tests and run `python -m unittest discover -s tests -v`.
3. Report software-only and MuJoCo results separately; identify checkpoints and dataset licences.
4. Avoid claims of hardware safety, successful real-VLA integrations or empirical improvements unless independently measured.

Pull requests should avoid committing datasets, large model weights, cached checkpoints or personal information.
