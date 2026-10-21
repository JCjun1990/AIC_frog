@echo off
rem Overnight job on the Windows laptop (README "交接说明", item 2): denser AI training data
rem (17 x 9 grid, 2 initial states per point, frog + sphere L5) -> retrain -> frog AI evaluation.
rem The PC stays awake while it runs (released afterwards). Resumable: run it again after an interruption.
rem Output: results\ai_generation_fine\   Log: results\ai_generation_fine_run.log (when started by the launcher)
cd /d "%~dp0..\.."
set PYTHONUTF8=1
.venv\Scripts\python scripts\ai_generation.py dataset --surfaces frog sphere5 --grid fine --seeds-per-point 2 --out results\ai_generation_fine --workers 20 || exit /b 1
.venv\Scripts\python scripts\ai_generation.py train --surfaces frog sphere5 --out results\ai_generation_fine || exit /b 1
if not exist results\ai_generation_fine\evaluation_jobs mkdir results\ai_generation_fine\evaluation_jobs
rem reuse the ordinary-pipeline runs of the earlier frog evaluation (same points, seeds and machine)
copy /y results\ai_generation\evaluation_jobs\pde_frog_*.pkl results\ai_generation_fine\evaluation_jobs\ >nul 2>&1
.venv\Scripts\python scripts\ai_generation.py evaluate --eval-surfaces frog --pipelines ai --out results\ai_generation_fine --eval-workers 20
echo overnight job finished
