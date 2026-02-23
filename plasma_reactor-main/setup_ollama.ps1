# PowerShell Setup Script for Ollama Integration with Plasma Control System

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Plasma Control System - Ollama LLM Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Ollama is installed
$ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue

if ($null -eq $ollamaPath) {
    Write-Host "WARNING: Ollama is not installed" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To use the Ollama integration:" -ForegroundColor Yellow
    Write-Host "1. Download Ollama from https://ollama.ai" -ForegroundColor Yellow
    Write-Host "2. Install and run Ollama" -ForegroundColor Yellow
    Write-Host "3. The system will automatically pull llama3.2" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Once installed:" -ForegroundColor Green
    Write-Host "  - Run: ollama serve" -ForegroundColor Green
    Write-Host "  - Run plasma control pipeline in another terminal" -ForegroundColor Green
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit
}

Write-Host "✓ Ollama found in PATH" -ForegroundColor Green
Write-Host ""

# Check if Ollama service is running
Write-Host "Checking if Ollama service is running..."

try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✓ Ollama service is running" -ForegroundColor Green
    Write-Host ""
    
    # Check if llama3.2 is available
    $models = $response.Content | ConvertFrom-Json
    $hasLlama = $models.models | Where-Object { $_.name -like "*llama3.2*" }
    
    if ($hasLlama) {
        Write-Host "✓ llama3.2 model already pulled" -ForegroundColor Green
        Write-Host ""
        Write-Host "================================================" -ForegroundColor Cyan
        Write-Host "Setup Complete!" -ForegroundColor Green
        Write-Host "================================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "You can now run:" -ForegroundColor Yellow
        Write-Host '  & "C:/Users/leela/Downloads/Telegram Desktop/plasma_reactor-main/.venv/Scripts/python.exe" run_complete_plasma_control.py' -ForegroundColor Green
        Write-Host ""
    } else {
        Write-Host "llama3.2 not found locally. Pulling now..." -ForegroundColor Yellow
        Write-Host ""
        
        & ollama pull llama3.2
        
        Write-Host ""
        Write-Host "================================================" -ForegroundColor Cyan
        Write-Host "✓ Setup Complete!" -ForegroundColor Green
        Write-Host "================================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "You can now run:" -ForegroundColor Yellow
        Write-Host '  & "C:/Users/leela/Downloads/Telegram Desktop/plasma_reactor-main/.venv/Scripts/python.exe" run_complete_plasma_control.py' -ForegroundColor Green
        Write-Host ""
        Write-Host "This will generate AI-powered reports using Llama 3.2" -ForegroundColor Cyan
        Write-Host ""
    }
} catch {
    Write-Host "⚠ Ollama service not running" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Start Ollama with:" -ForegroundColor Yellow
    Write-Host "  Command: ollama serve" -ForegroundColor Green
    Write-Host ""
    Write-Host "Then run the plasma control pipeline" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "The system will automatically pull llama3.2 on first run" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
