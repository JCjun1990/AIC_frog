@echo off
rem Lets the Mac reach this laptop's SSH server on the home (private) network.
rem Double-click this file, then click "Yes" when Windows asks for administrator rights.
net session >nul 2>&1 || (powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs" & exit /b)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetConnectionProfile | Set-NetConnectionProfile -NetworkCategory Private; Set-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -Enabled True -Action Allow -Profile Private,Domain; Get-NetConnectionProfile | ForEach-Object { Write-Host ('NETWORK: ' + $_.InterfaceAlias + ' -> ' + $_.NetworkCategory) }; Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' | ForEach-Object { Write-Host ('RULE: Enabled=' + $_.Enabled + ' Profile=' + $_.Profile) }; Get-CimInstance -Namespace root/SecurityCenter2 -ClassName FirewallProduct -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ('3RD-PARTY FIREWALL: ' + $_.displayName) }"
echo.
echo Done. You can close this window.
pause
