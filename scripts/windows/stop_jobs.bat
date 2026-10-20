@echo off
rem Stops every job of this project (Python started from turing\.venv, or running a turing script),
rem e.g. before gaming, so the CPU and GPU are free again. Nothing else on the PC is touched.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$root = (Resolve-Path '%~dp0..\..').Path; $jobs = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.ExecutablePath -like \"$root\*\" -or $_.CommandLine -like \"*$root\scripts\*\" }; if (-not $jobs) { Write-Host 'No turing jobs are running.' } else { $jobs | ForEach-Object { Write-Host ('Stopping PID ' + $_.ProcessId + ': ' + $_.CommandLine); taskkill /F /T /PID $_.ProcessId | Out-Null } }"
echo.
pause
