# Simple Pipeline Benchmark
Write-Host "`n=== PIPELINE PERFORMANCE BENCHMARK ===`n" -ForegroundColor Cyan

$testFile = "test\test_matmul_l3_tiling.mlir"
$iterations = 5

Write-Host "Test: $testFile" -ForegroundColor Yellow
Write-Host "Iterations: $iterations`n" -ForegroundColor Yellow

# Benchmark Optimized Pipeline
Write-Host "Benchmarking OPTIMIZED Pipeline..." -ForegroundColor Cyan
$optimizedTimes = @()
for ($i = 1; $i -le $iterations; $i++) {
    Write-Host "  Run $i..." -NoNewline
    $start = Get-Date
    .\build\bin\google-opt.exe $testFile --google-optimized-pipeline -o output\bench_opt.mlir 2>&1 | Out-Null
    $end = Get-Date
    $elapsed = ($end - $start).TotalMilliseconds
    $optimizedTimes += $elapsed
    Write-Host " $([math]::Round($elapsed, 2))ms" -ForegroundColor Gray
}

$avgOpt = ($optimizedTimes | Measure-Object -Average).Average
Write-Host "  Average: $([math]::Round($avgOpt, 2))ms`n" -ForegroundColor Green

# Benchmark Extreme Pipeline
Write-Host "Benchmarking EXTREME Pipeline (L3 Tiling)..." -ForegroundColor Cyan
$extremeTimes = @()
for ($i = 1; $i -le $iterations; $i++) {
    Write-Host "  Run $i..." -NoNewline
    $start = Get-Date
    .\build\bin\google-opt.exe $testFile --google-extreme-pipeline -o output\bench_ext.mlir 2>&1 | Out-Null
    $end = Get-Date
    $elapsed = ($end - $start).TotalMilliseconds
    $extremeTimes += $elapsed
    Write-Host " $([math]::Round($elapsed, 2))ms" -ForegroundColor Gray
}

$avgExt = ($extremeTimes | Measure-Object -Average).Average
Write-Host "  Average: $([math]::Round($avgExt, 2))ms`n" -ForegroundColor Green

# Results
Write-Host "=== RESULTS ===`n" -ForegroundColor Cyan
Write-Host "Optimized:  $([math]::Round($avgOpt, 2))ms" -ForegroundColor White
Write-Host "Extreme:    $([math]::Round($avgExt, 2))ms" -ForegroundColor White

$diff = $avgExt - $avgOpt
$pct = ($diff / $avgOpt) * 100
Write-Host "Difference: $([math]::Round($diff, 2))ms ($([math]::Round($pct, 1)) percent)`n" -ForegroundColor Yellow

# Output Analysis
if (Test-Path "output\bench_opt.mlir") {
    $optLoops = (Select-String "scf.for" "output\bench_opt.mlir" -AllMatches).Matches.Count
    Write-Host "Optimized loops: $optLoops" -ForegroundColor Gray
}

if (Test-Path "output\bench_ext.mlir") {
    $extLoops = (Select-String "scf.for" "output\bench_ext.mlir" -AllMatches).Matches.Count
    $extLLVM = (Select-String "llvm\." "output\bench_ext.mlir" -AllMatches).Matches.Count
    Write-Host "Extreme loops: $extLoops" -ForegroundColor Gray
    Write-Host "LLVM ops: $extLLVM" -ForegroundColor Gray
}

Write-Host "`nExtreme pipeline adds:" -ForegroundColor Yellow
Write-Host "  * L1+L2+L3 tiling" -ForegroundColor Gray
Write-Host "  * Affine optimization" -ForegroundColor Gray
Write-Host "  * Complete LLVM lowering" -ForegroundColor Gray
Write-Host "  * Expected 10-20x runtime speedup`n" -ForegroundColor Gray
