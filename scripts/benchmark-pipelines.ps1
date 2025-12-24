# Pipeline Performance Benchmark
# Compares Optimized vs Extreme (L3 Tiling) pipelines

Write-Host "`n🏁 PIPELINE PERFORMANCE BENCHMARK`n" -ForegroundColor Cyan
Write-Host "=" -NoNewline
for($i=0; $i -lt 70; $i++) { Write-Host "=" -NoNewline }
Write-Host "`n"

$testFile = "test\test_matmul_l3_tiling.mlir"
$iterations = 5

Write-Host "Test File: $testFile" -ForegroundColor Yellow
Write-Host "Iterations: $iterations`n" -ForegroundColor Yellow

# Benchmark Optimized Pipeline
Write-Host "📊 Benchmarking OPTIMIZED Pipeline..." -ForegroundColor Cyan
$optimizedTimes = @()
for ($i = 1; $i -le $iterations; $i++) {
    Write-Host "  Run $i/$iterations..." -NoNewline
    $start = Get-Date
    .\build\bin\google-opt.exe $testFile --google-optimized-pipeline -o output\bench_optimized.mlir 2>&1 | Out-Null
    $end = Get-Date
    $elapsed = ($end - $start).TotalMilliseconds
    $optimizedTimes += $elapsed
    Write-Host " $([math]::Round($elapsed, 2))ms" -ForegroundColor Gray
}

$avgOptimized = ($optimizedTimes | Measure-Object -Average).Average
$minOptimized = ($optimizedTimes | Measure-Object -Minimum).Minimum
$maxOptimized = ($optimizedTimes | Measure-Object -Maximum).Maximum

Write-Host "`n  Average: $([math]::Round($avgOptimized, 2))ms" -ForegroundColor Green
Write-Host "  Min: $([math]::Round($minOptimized, 2))ms" -ForegroundColor Gray
Write-Host "  Max: $([math]::Round($maxOptimized, 2))ms" -ForegroundColor Gray

# Benchmark Extreme Pipeline (L3 Tiling)
Write-Host "`n📊 Benchmarking EXTREME Pipeline (L3 Tiling)..." -ForegroundColor Cyan
$extremeTimes = @()
for ($i = 1; $i -le $iterations; $i++) {
    Write-Host "  Run $i/$iterations..." -NoNewline
    $start = Get-Date
    .\build\bin\google-opt.exe $testFile --google-extreme-pipeline -o output\bench_extreme.mlir 2>&1 | Out-Null
    $end = Get-Date
    $elapsed = ($end - $start).TotalMilliseconds
    $extremeTimes += $elapsed
    Write-Host " $([math]::Round($elapsed, 2))ms" -ForegroundColor Gray
}

$avgExtreme = ($extremeTimes | Measure-Object -Average).Average
$minExtreme = ($extremeTimes | Measure-Object -Minimum).Minimum
$maxExtreme = ($extremeTimes | Measure-Object -Maximum).Maximum

Write-Host "`n  Average: $([math]::Round($avgExtreme, 2))ms" -ForegroundColor Green
Write-Host "  Min: $([math]::Round($minExtreme, 2))ms" -ForegroundColor Gray
Write-Host "  Max: $([math]::Round($maxExtreme, 2))ms" -ForegroundColor Gray

# Analysis
Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline
for($i=0; $i -lt 70; $i++) { Write-Host "=" -NoNewline }
Write-Host "`n"

Write-Host "📈 ANALYSIS`n" -ForegroundColor Cyan

$difference = $avgExtreme - $avgOptimized
$percentDiff = ($difference / $avgOptimized) * 100

Write-Host "Compilation Time Comparison:" -ForegroundColor Yellow
Write-Host "  Optimized:  $([math]::Round($avgOptimized, 2))ms" -ForegroundColor White
Write-Host "  Extreme:    $([math]::Round($avgExtreme, 2))ms" -ForegroundColor White
$diffText = "$([math]::Round($difference, 2))ms ($([math]::Round($percentDiff, 1))percent)"
Write-Host "  Difference: $diffText" -ForegroundColor $(if ($difference -gt 0) {'Yellow'} else {'Green'})

Write-Host "`nOutput Analysis:" -ForegroundColor Yellow

if (Test-Path "output\bench_optimized.mlir") {
    $optLines = (Get-Content "output\bench_optimized.mlir").Count
    $optLoops = (Select-String "scf.for" "output\bench_optimized.mlir" -AllMatches).Matches.Count
    Write-Host "  Optimized: $optLines lines, $optLoops loops" -ForegroundColor White
}

if (Test-Path "output\bench_extreme.mlir") {
    $extLines = (Get-Content "output\bench_extreme.mlir").Count
    $extLoops = (Select-String "scf.for" "output\bench_extreme.mlir" -AllMatches).Matches.Count
    $extLLVM = (Select-String "llvm\." "output\bench_extreme.mlir" -AllMatches).Matches.Count
    Write-Host "  Extreme:   $extLines lines, $extLoops loops, $extLLVM LLVM ops" -ForegroundColor White
}

Write-Host "`n✅ Benchmark Complete!" -ForegroundColor Green
Write-Host "`nNote: Extreme pipeline includes:" -ForegroundColor Gray
Write-Host "  + L1+L2+L3 tiling (3 level cache hierarchy)" -ForegroundColor Gray
Write-Host "  + Affine loop optimization" -ForegroundColor Gray
Write-Host "  + Complete LLVM lowering" -ForegroundColor Gray
Write-Host "  + Expected runtime speedup: 10 to 20x" -ForegroundColor Gray
