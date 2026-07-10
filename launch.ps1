param(
  [int]$Port = 4178,
  [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$Root = [System.IO.Path]::GetFullPath($PSScriptRoot)
$RootPrefix = $Root.TrimEnd([char[]]@([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)) + [System.IO.Path]::DirectorySeparatorChar
$Address = [System.Net.IPAddress]::Parse('127.0.0.1')
$Listener = [System.Net.Sockets.TcpListener]::new($Address, $Port)

$MimeTypes = @{
  '.html' = 'text/html; charset=utf-8'
  '.css'  = 'text/css; charset=utf-8'
  '.js'   = 'text/javascript; charset=utf-8'
  '.mjs'  = 'text/javascript; charset=utf-8'
  '.json' = 'application/json; charset=utf-8'
  '.md'   = 'text/markdown; charset=utf-8'
  '.png'  = 'image/png'
  '.jpg'  = 'image/jpeg'
  '.jpeg' = 'image/jpeg'
  '.webp' = 'image/webp'
  '.svg'  = 'image/svg+xml'
}

function Write-HttpResponse {
  param(
    [System.Net.Sockets.NetworkStream]$Stream,
    [int]$Status,
    [string]$StatusText,
    [byte[]]$Body,
    [string]$ContentType,
    [bool]$HeadOnly = $false
  )

  $Header = "HTTP/1.1 $Status $StatusText`r`nContent-Type: $ContentType`r`nContent-Length: $($Body.Length)`r`nCache-Control: no-store`r`nAllow: GET, HEAD`r`nX-Content-Type-Options: nosniff`r`nReferrer-Policy: no-referrer`r`nCross-Origin-Opener-Policy: same-origin`r`nCross-Origin-Resource-Policy: same-origin`r`nContent-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'`r`nConnection: close`r`n`r`n"
  $HeaderBytes = [System.Text.Encoding]::ASCII.GetBytes($Header)
  $Stream.Write($HeaderBytes, 0, $HeaderBytes.Length)
  if (-not $HeadOnly -and $Body.Length -gt 0) {
    $Stream.Write($Body, 0, $Body.Length)
  }
}

try {
  $Listener.Start()
  $Url = "http://127.0.0.1:$Port/"
  Write-Host "Butter Lab BL-05 is running at $Url" -ForegroundColor Yellow
  Write-Host 'Keep this window open. Press Ctrl+C to stop the lab.' -ForegroundColor DarkGray
  if (-not $NoBrowser) {
    $BrowserCandidates = @(
      'C:\Program Files\Google\Chrome\Application\chrome.exe',
      'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
      'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
      'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
    )
    $Browser = $BrowserCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if ($Browser) {
      Start-Process -FilePath $Browser -ArgumentList @('--new-window', $Url)
    }
    else {
      Start-Process $Url
    }
  }

  while ($true) {
    $Client = $Listener.AcceptTcpClient()
    try {
      $Client.ReceiveTimeout = 5000
      $Client.SendTimeout = 5000
      $Stream = $Client.GetStream()
      $Stream.ReadTimeout = 5000
      $Stream.WriteTimeout = 5000
      $Reader = [System.IO.StreamReader]::new($Stream, [System.Text.Encoding]::ASCII, $false, 4096, $true)
      $RequestLine = $Reader.ReadLine()
      if ([string]::IsNullOrWhiteSpace($RequestLine)) { continue }

      $Parts = $RequestLine.Split(' ')
      $Method = $Parts[0]
      $RawPath = if ($Parts.Length -gt 1) { $Parts[1].Split('?')[0] } else { '/' }
      $Headers = @{}
      while ($true) {
        $Line = $Reader.ReadLine()
        if ([string]::IsNullOrEmpty($Line)) { break }
        $Separator = $Line.IndexOf(':')
        if ($Separator -gt 0) {
          $Headers[$Line.Substring(0, $Separator).Trim()] = $Line.Substring($Separator + 1).Trim()
        }
      }

      $AllowedHosts = @("127.0.0.1:$Port", "localhost:$Port")
      if (-not $Headers.ContainsKey('Host') -or $AllowedHosts -notcontains $Headers['Host']) {
        $Body = [System.Text.Encoding]::UTF8.GetBytes('Forbidden host')
        Write-HttpResponse -Stream $Stream -Status 403 -StatusText 'Forbidden' -Body $Body -ContentType 'text/plain; charset=utf-8' -HeadOnly ($Method -eq 'HEAD')
        continue
      }

      if ($Method -ne 'GET' -and $Method -ne 'HEAD') {
        $Body = [System.Text.Encoding]::UTF8.GetBytes('Method not allowed')
        Write-HttpResponse -Stream $Stream -Status 405 -StatusText 'Method Not Allowed' -Body $Body -ContentType 'text/plain; charset=utf-8'
        continue
      }

      try {
        $DecodedPath = [System.Uri]::UnescapeDataString($RawPath.TrimStart('/')).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
        if ([string]::IsNullOrWhiteSpace($DecodedPath)) { $DecodedPath = 'index.html' }
        $Candidate = [System.IO.Path]::GetFullPath((Join-Path $Root $DecodedPath))
      }
      catch {
        $Body = [System.Text.Encoding]::UTF8.GetBytes('Bad request path')
        Write-HttpResponse -Stream $Stream -Status 400 -StatusText 'Bad Request' -Body $Body -ContentType 'text/plain; charset=utf-8' -HeadOnly ($Method -eq 'HEAD')
        continue
      }
      if (-not $Candidate.StartsWith($RootPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        $Body = [System.Text.Encoding]::UTF8.GetBytes('Not found')
        Write-HttpResponse -Stream $Stream -Status 404 -StatusText 'Not Found' -Body $Body -ContentType 'text/plain; charset=utf-8' -HeadOnly ($Method -eq 'HEAD')
        continue
      }

      $Body = [System.IO.File]::ReadAllBytes($Candidate)
      $Extension = [System.IO.Path]::GetExtension($Candidate).ToLowerInvariant()
      $ContentType = if ($MimeTypes.ContainsKey($Extension)) { $MimeTypes[$Extension] } else { 'application/octet-stream' }
      Write-HttpResponse -Stream $Stream -Status 200 -StatusText 'OK' -Body $Body -ContentType $ContentType -HeadOnly ($Method -eq 'HEAD')
    }
    catch {
      Write-Warning $_.Exception.Message
    }
    finally {
      $Client.Close()
    }
  }
}
finally {
  $Listener.Stop()
}
