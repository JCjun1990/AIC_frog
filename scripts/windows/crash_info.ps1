# Read-only crash report for the RTX 4080 laptop: last boot, blue screens (stop codes), unexpected
# shutdowns, hardware (WHEA) errors, GPU driver resets, warnings just before the last boot, and the
# BIOS / microcode / driver versions. Changes nothing on the PC.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\crash_info.ps1
[Console]::OutputEncoding = [Text.Encoding]::UTF8
function Short($s, $n = 220) { $s = ("$s" -replace '\s+', ' ').Trim(); if ($s.Length -gt $n) { $s.Substring(0, $n) + '...' } else { $s } }

$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$bios = Get-CimInstance Win32_BIOS
$rev = (Get-ItemProperty 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0').'Update Revision'
$mc = if ($rev -is [byte[]] -and $rev.Length -ge 4) { '0x{0:X}' -f [BitConverter]::ToUInt32($rev, $rev.Length - 4) } else { "$rev" }
"Last boot: {0}   now: {1}   Windows build {2}" -f $os.LastBootUpTime, (Get-Date), $os.BuildNumber
"Model: {0} {1}   RAM: {2:N1} GB" -f $cs.Manufacturer, $cs.Model, ($cs.TotalPhysicalMemory / 1GB)
"CPU: {0}   microcode: {1}" -f $cpu.Name.Trim(), $mc
"BIOS: {0} {1} ({2:yyyy-MM-dd})" -f $bios.Manufacturer, $bios.SMBIOSBIOSVersion, $bios.ReleaseDate

"`n== Blue screens (WER-SystemErrorReporting 1001), newest first"
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WER-SystemErrorReporting'} -MaxEvents 15 -ErrorAction SilentlyContinue |
    ForEach-Object { "{0}  stop code {1}" -f $_.TimeCreated, $_.Properties[0].Value }

"`n== Unexpected resets (Kernel-Power 41; BugcheckCode 0 = hard reset / power loss), newest first"
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'; Id=41} -MaxEvents 15 -ErrorAction SilentlyContinue |
    ForEach-Object { "{0}  BugcheckCode=0x{1:X}" -f $_.TimeCreated, $_.Properties[0].Value }

"`n== Hardware errors (WHEA-Logger), newest first"
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'} -MaxEvents 15 -ErrorAction SilentlyContinue |
    ForEach-Object { "{0}  id {1}: {2}" -f $_.TimeCreated, $_.Id, (Short $_.Message 260) }

"`n== GPU driver resets (Display 4101, nvlddmkm), last 90 days"
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Display','nvlddmkm'; StartTime=(Get-Date).AddDays(-90)} -MaxEvents 10 -ErrorAction SilentlyContinue |
    ForEach-Object { "{0}  {1} ({2}): {3}" -f $_.TimeCreated, $_.ProviderName, $_.Id, (Short $_.Message 160) }

"`n== Warnings and errors in the 15 minutes before the last boot"
$t1 = $os.LastBootUpTime; $t0 = $t1.AddMinutes(-15)
foreach ($log in 'System', 'Application') {
    Get-WinEvent -FilterHashtable @{LogName=$log; Level=1,2,3; StartTime=$t0; EndTime=$t1} -ErrorAction SilentlyContinue |
        Select-Object -First 20 | ForEach-Object { "{0}  [{1}] {2} ({3}): {4}" -f $_.TimeCreated, $log, $_.ProviderName, $_.Id, (Short $_.Message 150) }
}

"`n== Display, network and storage drivers"
Get-CimInstance Win32_PnPSignedDriver | Where-Object { $_.DeviceClass -in 'DISPLAY','NET','SCSIADAPTER','HDC' -and $_.DeviceName -and $_.DeviceName -notlike 'WAN Miniport*' } |
    Sort-Object DeviceClass, DeviceName | ForEach-Object { "{0,-12} {1,-58} {2,-18} {3:yyyy-MM-dd}" -f $_.DeviceClass, (Short $_.DeviceName 58), $_.DriverVersion, $_.DriverDate }

"`n== File-system minifilters (fltmc; FLTMGR.SYS is on the stack of the recent crashes)"
fltmc filters 2>&1 | ForEach-Object { "$_" } | Where-Object { $_.Trim() }

"`n== Running kernel drivers not made by Microsoft"
Get-CimInstance Win32_SystemDriver -Filter "State='Running'" | ForEach-Object {
    $p = $_.PathName -replace '^\\\?\?\\', '' -replace '^\\SystemRoot\\', "$env:SystemRoot\" -replace '^system32\\', "$env:SystemRoot\System32\"
    $vi = (Get-Item -LiteralPath $p -ErrorAction SilentlyContinue).VersionInfo
    if ($vi -and $vi.CompanyName -notmatch 'Microsoft') {
        "{0,-24} {1,-34} {2}" -f $_.Name, (Short $vi.CompanyName 34), (Short $vi.FileDescription 70) }
}

"`n== Dump files"
Get-ChildItem C:\Windows\Minidump, C:\Windows\MEMORY.DMP, C:\Windows\LiveKernelReports -Recurse -ErrorAction SilentlyContinue |
    Where-Object { -not $_.PSIsContainer } | Sort-Object LastWriteTime -Descending | Select-Object -First 8 |
    ForEach-Object { "{0}  {1,15:N0} B  {2}" -f $_.LastWriteTime, $_.Length, $_.FullName }
