# Import the validated v0.5 source archive

The GitHub integration has uploaded documentation and packaging metadata, but has not yet transferred the executable source tree. Do not run `pip install -e .` on this branch until the `src/` and `tests/` folders are imported.

Download the `CausalForge-v0.5-source.zip` attachment from the ChatGPT conversation to your computer.

## Windows PowerShell

```powershell
git clone https://github.com/aaflakiyan/CausalForge.git
$repo = (Resolve-Path .\CausalForge).Path
$staging = Join-Path $env:TEMP "causalforge-import-v05"
New-Item -ItemType Directory -Force -Path $staging | Out-Null
Expand-Archive -LiteralPath "$HOME\Downloads\CausalForge-v0.5-source.zip" -DestinationPath $staging -Force
robocopy "$staging\CausalForge" "$repo" /E /XD .git
cd $repo
git add -A
git commit -m "Import validated CausalForge v0.5 source and tests"
git push origin main
```

Robocopy exit codes 0–7 normally denote success or nonfatal differences. Check output if it reports errors.

The original local package passed 43 tests with seven MuJoCo-specific tests skipped. Run `python -m unittest discover -s tests -v` after importing. GitHub Actions should then run Python and MuJoCo tests on GitHub-hosted runners; these test results are not yet verified.

Do not commit trained model weights, credentials, clinical data, or private robot logs.
