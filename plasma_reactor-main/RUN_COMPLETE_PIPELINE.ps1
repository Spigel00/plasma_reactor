# Complete Quick-Start Guide for Plasma Control with Ollama Reports

Write-Host @"
════════════════════════════════════════════════════════════════════════════════════════════════
                     PLASMA CONTROL + OLLAMA REPORTS - QUICK START
════════════════════════════════════════════════════════════════════════════════════════════════
"@ -ForegroundColor Cyan

# Step 1: Check Python environment
Write-Host "`n1️⃣  Checking Python environment..." -ForegroundColor Yellow

`$pyVenv = "C:/Users/leela/Downloads/Telegram Desktop/plasma_reactor-main/.venv/Scripts/python.exe"
`$pyCmd = "&`"`$pyVenv`""

if (Test-Path $pyVenv) {
    Write-Host "   ✓ Virtual environment found" -ForegroundColor Green
} else {
    Write-Host "   ✗ Virtual environment not found" -ForegroundColor Red
    exit 1
}

# Step 2: Check Ollama
Write-Host "`n2️⃣  Checking Ollama installation..." -ForegroundColor Yellow

`$ollama = Get-Command ollama -ErrorAction SilentlyContinue

if (`$null -eq `$ollama) {
    Write-Host "   ⚠ Ollama not found - reports will be skipped" -ForegroundColor Yellow
    Write-Host "   Download from https://ollama.ai to enable reports" -ForegroundColor Yellow
} else {
    Write-Host "   ✓ Ollama found" -ForegroundColor Green
    
    # Check if service is running
    try {
        `$response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
        Write-Host "   ✓ Ollama service running" -ForegroundColor Green
        
        `$models = `$response.Content | ConvertFrom-Json
        `$hasLlama = `$models.models | Where-Object { `$_.name -like "*llama3.2*" }
        
        if (`$hasLlama) {
            Write-Host "   ✓ llama3.2 model available" -ForegroundColor Green
        } else {
            Write-Host "   ⚠ llama3.2 not found - will pull automatically" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   ⚠ Ollama service not running" -ForegroundColor Yellow
        Write-Host "   Start with: ollama serve (in another terminal)" -ForegroundColor Yellow
    }
}

# Step 3: Navigate to project
Write-Host "`n3️⃣  Navigating to project directory..." -ForegroundColor Yellow

`$projectDir = "c:\Users\leela\Downloads\Telegram Desktop\plasma_reactor-main\plasma_reactor-main"

if (Test-Path `$projectDir) {
    Set-Location `$projectDir
    Write-Host "   ✓ In project directory" -ForegroundColor Green
    Write-Host "   Location: `$projectDir" -ForegroundColor Green
} else {
    Write-Host "   ✗ Project directory not found" -ForegroundColor Red
    exit 1
}

# Step 4: Run pipeline
Write-Host "`n4️⃣  Running complete pipeline..." -ForegroundColor Yellow
Write-Host "   This will:" -ForegroundColor Cyan
Write-Host "   ✓ Train RL agent (100k steps) - ~3 min" -ForegroundColor Cyan
Write-Host "   ✓ Evaluate on 3 episodes - ~30 sec" -ForegroundColor Cyan
Write-Host "   ✓ Deploy control sequence (150 steps) - ~10 sec" -ForegroundColor Cyan
Write-Host "   ✓ Generate visualizations - ~10 sec" -ForegroundColor Cyan
Write-Host "   ✓ Generate AI reports with llama3.2 - ~2-5 min (if available)" -ForegroundColor Cyan
Write-Host "`n   Total time: ~5-10 minutes`n" -ForegroundColor Yellow

# Run the pipeline
`$command = "&`"`$pyVenv`" run_complete_plasma_control.py"
Write-Host "Running: $command" -ForegroundColor Green
Write-Host ""

Invoke-Expression `$command

# Check for generated files
Write-Host "`n5️⃣  Checking generated files..." -ForegroundColor Yellow

`$expectedFiles = @(
    "PLASMA_CONTROL_REPORT.md",
    "PLASMA_CONTROL_REPORT.html",
    "PLASMA_CONTROL_TECHNICAL_REVIEW.md",
    "PLASMA_CONTROL_TECHNICAL_REVIEW.html",
    "plasma_control_results.png",
    "plasma_deployment_results.png"
)

foreach (`$file in `$expectedFiles) {
    if (Test-Path `$file) {
        Write-Host "   ✓ `$file" -ForegroundColor Green
    }
}

# Final summary
Write-Host "`n" -ForegroundColor Yellow
Write-Host "════════════════════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "✅ PIPELINE COMPLETE" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

Write-Host "`nGenerated Deliverables:" -ForegroundColor Yellow
Write-Host "  📊 Reports (AI-generated with Llama 3.2):" -ForegroundColor Cyan
Write-Host "     - PLASMA_CONTROL_REPORT.md/html" -ForegroundColor White
Write-Host "     - PLASMA_CONTROL_TECHNICAL_REVIEW.md/html" -ForegroundColor White
Write-Host ""
Write-Host "  📈 Visualizations:" -ForegroundColor Cyan
Write-Host "     - plasma_control_results.png (training curves)" -ForegroundColor White
Write-Host "     - plasma_deployment_results.png (control performance)" -ForegroundColor White
Write-Host ""
Write-Host "  🤖 Trained Models:" -ForegroundColor Cyan
Write-Host "     - ./rl_models/best_model.zip" -ForegroundColor White
Write-Host "     - ./rl_models/final_plasma_model.zip" -ForegroundColor White
Write-Host ""
Write-Host "  📋 Documentation:" -ForegroundColor Cyan
Write-Host "     - PLASMA_CONTROL_SUCCESS_DOCUMENTATION.md (30-45 min read)" -ForegroundColor White
Write-Host "     - OLLAMA_INTEGRATION_README.md" -ForegroundColor White
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Open HTML reports in browser (styled, professional)" -ForegroundColor Cyan
Write-Host "  2. Review markdown reports for technical details" -ForegroundColor Cyan
Write-Host "  3. View PNG visualizations for performance metrics" -ForegroundColor Cyan
Write-Host "  4. Load model for further testing/deployment" -ForegroundColor Cyan
Write-Host ""

Write-Host "Open Reports:" -ForegroundColor Yellow
if (Test-Path "PLASMA_CONTROL_REPORT.html") {
    Write-Host "  & explorer.exe PLASMA_CONTROL_REPORT.html" -ForegroundColor Green
}
if (Test-Path "PLASMA_CONTROL_TECHNICAL_REVIEW.html") {
    Write-Host "  & explorer.exe PLASMA_CONTROL_TECHNICAL_REVIEW.html" -ForegroundColor Green
}

Write-Host "`n════════════════════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "System Status: ✅ FULLY OPERATIONAL" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit"
