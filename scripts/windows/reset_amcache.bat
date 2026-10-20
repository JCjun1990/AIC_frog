@echo off
rem Resets Windows' application-inventory hive (Amcache.hve): every registry-related blue screen on this
rem laptop was checking it when it crashed (see README). Double-click, click "Yes" for administrator
rem rights, then restart. The old files go to a dated backup folder; Windows creates a fresh hive itself.
net session >nul 2>&1 || (powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs" & exit /b)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0reset_amcache.ps1"
echo.
pause
