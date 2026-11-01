param(
    [string[]]$Languages = @("osd", "eng", "hin", "ben", "guj", "kan", "mar", "pan", "tam", "tel")
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$tessdataPath = Join-Path $projectRoot "data\ocr\tessdata"
$supportedLanguages = @("osd", "eng", "hin", "ben", "tam", "tel", "mar", "guj", "kan", "pan")

$invalidLanguages = @($Languages | Where-Object { $_ -notin $supportedLanguages })
if ($invalidLanguages.Count -gt 0) {
    throw "Unsupported Tesseract language code(s): $($invalidLanguages -join ', ')"
}

New-Item -ItemType Directory -Force -Path $tessdataPath | Out-Null
$baseUrl = "https://github.com/tesseract-ocr/tessdata_fast/raw/refs/heads/main"

foreach ($language in ($Languages | Select-Object -Unique)) {
    $target = Join-Path $tessdataPath "$language.traineddata"
    if (Test-Path -LiteralPath $target) {
        Write-Output "$language model already exists."
        continue
    }

    $temporary = "$target.download"
    try {
        Invoke-WebRequest -Uri "$baseUrl/$language.traineddata" -OutFile $temporary
        if ((Get-Item -LiteralPath $temporary).Length -lt 10000) {
            throw "Downloaded model file is unexpectedly small."
        }
        Move-Item -LiteralPath $temporary -Destination $target
        Write-Output "Installed local OCR model: $language"
    }
    finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

Write-Output "Tesseract language data is in $tessdataPath"
