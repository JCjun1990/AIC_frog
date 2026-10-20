# One-time setup on the Windows RTX 4080 laptop: lets the team Mac connect over SSH with its key.
# Run in an elevated PowerShell ("Terminal (Admin)"). Safe to run more than once.

# 1) Install and start the OpenSSH server
$cap = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
if ($cap.State -ne 'Installed') { Add-WindowsCapability -Online -Name $cap.Name | Out-Null }
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd

# 2) Allow inbound SSH (TCP 22) through the firewall
if (Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue) {
    Enable-NetFirewallRule -Name 'OpenSSH-Server-In-TCP'
} else {
    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' -Enabled True `
        -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
}

# 3) Authorise the Mac's public key (both places: admin accounts use the first, others the second)
$key = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHyK7vr2rqb8fRER0/q3TQgyNmoo+m3FRRsfEGfvp6gZ mac-to-4080'
$admKeys = "$env:ProgramData\ssh\administrators_authorized_keys"
if (-not (Test-Path $admKeys) -or -not (Select-String -Path $admKeys -SimpleMatch $key -Quiet)) { Add-Content -Force -Path $admKeys -Value $key }
icacls.exe $admKeys /inheritance:r /grant "*S-1-5-32-544:F" /grant "*S-1-5-18:F" | Out-Null
New-Item -ItemType Directory -Force "$env:USERPROFILE\.ssh" | Out-Null
$userKeys = "$env:USERPROFILE\.ssh\authorized_keys"
if (-not (Test-Path $userKeys) -or -not (Select-String -Path $userKeys -SimpleMatch $key -Quiet)) { Add-Content -Force -Path $userKeys -Value $key }

# 4) Report what the Mac needs
Write-Host ""
Write-Host "USERNAME: $env:USERNAME"
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
    ForEach-Object { Write-Host "IP: $($_.IPAddress) ($($_.InterfaceAlias))" }
Write-Host "sshd: $((Get-Service sshd).Status)"
