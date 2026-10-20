# Moves Windows' application-inventory hive (Amcache.hve) and its log / transaction files into a dated
# backup folder, so Windows creates a fresh one. Every registry-related blue screen on this laptop
# (2026-09 to 2026-10) crashed while the kernel was checking this hive; see README.
# Run elevated (reset_amcache.bat does that). Files that are in use are moved at the next restart.
# Undo: move the files from the backup folder back into C:\Windows\appcompat\Programs.
$ErrorActionPreference = 'Stop'
$src = Join-Path $env:SystemRoot 'appcompat\Programs'
$files = Get-ChildItem -Force -LiteralPath $src -Filter 'Amcache.hve*' | Where-Object { -not $_.PSIsContainer }
if (-not $files) { Write-Host 'No Amcache.hve files found - nothing to do.'; return }
$dst = Join-Path $src ('Amcache-backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $dst | Out-Null
Add-Type -Namespace Win32 -Name Native -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
public static extern bool MoveFileEx(string from, string to, int flags);
'@
Stop-Service PcaSvc -Force -ErrorAction SilentlyContinue      # Program Compatibility Assistant uses the hive
$pending = 0
foreach ($f in $files) {
    $to = Join-Path $dst $f.Name
    try {
        Move-Item -LiteralPath $f.FullName -Destination $to
        Write-Host "moved              $($f.Name)"
    } catch {
        if ([Win32.Native]::MoveFileEx($f.FullName, $to, 4)) {      # 4 = MOVEFILE_DELAY_UNTIL_REBOOT
            $pending++
            Write-Host "moved at restart   $($f.Name) (in use now)"
        } else {
            Write-Host "FAILED             $($f.Name): $($_.Exception.Message)"
        }
    }
}
Start-Service PcaSvc -ErrorAction SilentlyContinue
Write-Host ''
Write-Host "Backup folder: $dst"
if ($pending) { Write-Host "$pending file(s) are in use; they move when Windows restarts. Please restart now." }
else { Write-Host 'Done. Windows creates a fresh Amcache.hve by itself; a restart is still a good idea.' }
