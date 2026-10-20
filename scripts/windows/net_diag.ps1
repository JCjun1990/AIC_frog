# Read-only network check for the RTX 4080 laptop: adapters, addresses, gateway, DNS, reachability with
# and without the system proxy, and whether the proxy program is actually listening. Changes nothing.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\net_diag.ps1
[Console]::OutputEncoding = [Text.Encoding]::UTF8

"== Adapters that are up"
Get-NetAdapter | Where-Object Status -eq Up | ForEach-Object { "  {0} | {1} | {2}" -f $_.Name, $_.InterfaceDescription, $_.LinkSpeed }

"`n== Connection profiles (what the taskbar icon reports)"
Get-NetConnectionProfile | ForEach-Object { "  {0}: {1}, IPv4 {2}, IPv6 {3}" -f $_.InterfaceAlias, $_.NetworkCategory, $_.IPv4Connectivity, $_.IPv6Connectivity }

"`n== IPv4 configuration"
Get-NetIPConfiguration | Where-Object { $_.NetAdapter.Status -eq 'Up' } | ForEach-Object {
    $dns = ($_.DNSServer | Where-Object AddressFamily -eq 2 | ForEach-Object { $_.ServerAddresses }) -join ','
    "  {0}: IP {1}  gateway {2}  DNS {3}" -f $_.InterfaceAlias, ($_.IPv4Address.IPAddress -join ','), ($_.IPv4DefaultGateway.NextHop -join ','), $dns }

"`n== Default routes"
Get-NetRoute -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue |
    ForEach-Object { "  {0} via {1} (metric {2}+{3})" -f $_.InterfaceAlias, $_.NextHop, $_.RouteMetric, $_.InterfaceMetric }

"`n== Reachability"
$gw = (Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | Select-Object -First 1).IPv4DefaultGateway.NextHop
foreach ($h in @($gw, '223.5.5.5', '114.114.114.114')) { if ($h) { "  ping {0,-16} {1}" -f $h, (Test-Connection -ComputerName $h -Count 2 -Quiet) } }
foreach ($n in 'www.baidu.com', 'www.msftconnecttest.com', 'pypi.org') {
    try { $a = (Resolve-DnsName $n -Type A -ErrorAction Stop | Where-Object Type -eq 'A' | Select-Object -First 1).IPAddress; "  DNS {0,-24} {1}" -f $n, $a }
    catch { "  DNS {0,-24} FAIL: {1}" -f $n, $_.Exception.Message } }

"`n== HTTP without any proxy"
foreach ($u in 'http://www.msftconnecttest.com/connecttest.txt', 'https://www.baidu.com') {
    $o = & curl.exe -s -o NUL -m 8 --noproxy '*' -w '%{http_code} in %{time_total}s' $u 2>&1
    "  {0,-50} {1}" -f $u, $o }

"`n== System proxy"
$is = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings'
"  enabled={0}  server={1}  PAC={2}" -f $is.ProxyEnable, $is.ProxyServer, $is.AutoConfigURL
if ($is.ProxyEnable -and $is.ProxyServer) {
    $hp = ($is.ProxyServer -split ';')[0] -replace '^[a-z]+=', ''
    $port = [int]($hp -split ':')[-1]
    $l = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($l) { "  port {0} is served by {1} (pid {2})" -f $port, (Get-Process -Id $l.OwningProcess).ProcessName, $l.OwningProcess }
    else { "  port {0}: NOTHING is listening, so every app that uses the system proxy has no internet" -f $port }
    $o = & curl.exe -s -o NUL -m 8 -x "http://$hp" -w '%{http_code} in %{time_total}s' 'https://www.baidu.com' 2>&1
    "  https://www.baidu.com via {0}: {1}" -f $hp, $o
}

"`n== Proxy / VPN / accelerator programs running"
$p = Get-Process | Where-Object { $_.ProcessName -match 'clash|mihomo|verge|v2ray|xray|sing-box|nekoray|shadowsocks|trojan|proxifier|hiddify|flclash|leigod|xunyou|uuacc|netease' }
if ($p) { $p | ForEach-Object { "  {0} (pid {1}) {2}" -f $_.ProcessName, $_.Id, $_.Path } } else { '  none' }
