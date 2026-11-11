# PowerShell script chay nhanh ADAS Road Assistant
param(
    [string]$Source = "cao_toc",
    [double]$Speed = 90
)

$Aliases = @{
    "hem_ngo" = "data/videos/demo_hem_ngo.mp4"
    "cao_toc" = "data/videos/demo_cao_toc.mp4"
    "duong_doi" = "data/videos/demo_duong_doi_do_thi.mp4"
    "duong_doi_do_thi" = "data/videos/demo_duong_doi_do_thi.mp4"
    "pho_hai_chieu" = "data/videos/demo_pho_hai_chieu.mp4"
    "pho_2chieu" = "data/videos/demo_pho_hai_chieu.mp4"
    "ngoai_khu_dan_cu" = "data/videos/demo_ngoai_khu_dan_cu.mp4"
    "ngoai_kdc" = "data/videos/demo_ngoai_khu_dan_cu.mp4"
}

if ($Aliases.ContainsKey($Source)) {
    $VideoPath = $Aliases[$Source]
} else {
    $VideoPath = $Source
}

$VenvPython = Resolve-Path "$PSScriptRoot\.venv\Scripts\python.exe" -ErrorAction SilentlyContinue
if (-not $VenvPython) {
    $VenvPython = Resolve-Path "$PSScriptRoot\..\.venv\Scripts\python.exe" -ErrorAction SilentlyContinue
}
if (-not $VenvPython) {
    $VenvPython = "python"
}

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "    KHOI DONG ADAS ROAD ASSISTANT (4 TRANG THAI CANH BAO + GIAI THICH)" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "▶ Video nguon : $VideoPath" -ForegroundColor Yellow
Write-Host "▶ Toc do xe   : $Speed km/h (Bam phim +/- tren ban phim de tang giam)" -ForegroundColor Green
Write-Host "▶ 4 Trang thai: Dang xac dinh | An toan | Chu y | Vuot toc`n" -ForegroundColor Magenta

& $VenvPython -m adas --source "$VideoPath" --demo-speed $Speed $args
