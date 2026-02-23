# Plasma Control System - Ollama LLM Integration

This system now integrates **Ollama Llama 3.2** to generate AI-powered reports and analysis of plasma control results.

## What's New

The plasma control pipeline now includes **Step 5: AI-powered report generation** using Ollama's Llama 3.2 LLM:

- **Executive Report** (MD & HTML): High-level summary, achievements, and analysis
- **Technical Review** (MD & HTML): Deep technical analysis, problem solving, and recommendations

## Installation

### Step 1: Install Ollama

1. Download Ollama from: https://ollama.ai
2. Install on your system
3. Run: `ollama serve` (in one terminal)

### Step 2: Setup Ollama Integration

Run the setup script (choose one):

**PowerShell:**
```powershell
& "C:/Users/leela/Downloads/Telegram Desktop/plasma_reactor-main/plasma_reactor-main/setup_ollama.ps1"
```

**Command Prompt:**
```cmd
setup_ollama.bat
```

This will automatically pull the `llama3.2` model (~4.1 GB).

### Step 3: Run Pipeline with Report Generation

```powershell
cd "c:\Users\leela\Downloads\Telegram Desktop\plasma_reactor-main\plasma_reactor-main"
& "C:/Users/leela/Downloads/Telegram Desktop/plasma_reactor-main/.venv/Scripts/python.exe" run_complete_plasma_control.py
```

The pipeline will:
1. Train RL agent (100k timesteps)
2. Evaluate on 3 episodes
3. Deploy in real-time control
4. **Generate AI reports with Llama 3.2**
5. Save as MD and HTML files

## Generated Files

### Reports Generated

After running the pipeline, you'll get 4 new files:

1. **PLASMA_CONTROL_REPORT.md** - Executive report in Markdown
2. **PLASMA_CONTROL_REPORT.html** - Executive report in HTML (styled)
3. **PLASMA_CONTROL_TECHNICAL_REVIEW.md** - Technical review in Markdown
4. **PLASMA_CONTROL_TECHNICAL_REVIEW.html** - Technical review in HTML (styled)

## Features

### Executive Report
- Summary of plasma control achievements
- System architecture overview
- Key performance highlights
- Technical analysis
- Safety validation results
- Deployment readiness assessment
- Recommendations for future work

### Technical Review
- Problem identification and root cause analysis
- Solution effectiveness assessment
- Convergence and consistency analysis
- Safety mechanism validation
- Production readiness evaluation
- Mitigation strategies and conclusions

## Files

### Python Modules

**`ollama_report_generator.py`** (Main integration file)
- `OllamaReportGenerator` class: Manages Ollama connection and model
- `generate_plasma_report()`: Creates executive report using Llama 3.2
- `generate_technical_review()`: Creates technical review using Llama 3.2
- Format converters: Markdown and HTML output functions
- Automatic model pulling if not available

### Setup Scripts

**`setup_ollama.ps1`** - PowerShell setup script
- Checks Ollama installation
- Verifies service is running
- Pulls llama3.2 model
- Provides setup instructions

**`setup_ollama.bat`** - Command Prompt setup script
- Windows batch version of setup

### Integration

**`run_complete_plasma_control.py`** (Updated)
- Added: `generate_llm_reports()` function
- Integrated report generation as Step 5
- Reports happen after deployment
- Graceful fallback if Ollama unavailable

## System Requirements

- **Ollama**: https://ollama.ai (must be running)
- **llama3.2 model**: ~4.1 GB disk space
- **RAM**: 8+ GB recommended
- **Network**: Internet for first model download

## Troubleshooting

### "Ollama not running"
```powershell
# Start Ollama service
ollama serve
# In another terminal, run the pipeline
```

### "Connection refused"
- Ensure `ollama serve` is running
- Check that port 11434 is not blocked
- Wait 5 seconds after starting Ollama

### "Model not found"
```powershell
# Manually pull the model
ollama pull llama3.2
```

### Reports not generating
- Keep it simple: the pipeline works great without reports
- Reports are optional; control system works independently
- Check Python logs for Ollama connection errors

## Report Customization

Edit `ollama_report_generator.py` to customize:

**Change model:**
```python
ollama = OllamaReportGenerator(model_name="llama2")  # or another model
```

**Adjust report style:**
```python
def generate_plasma_report(self, control_data):
    # Modify the prompt to change report style
    prompt = f"""Your custom prompt here..."""
```

**Change output format:**
```python
# Add PDF support
# Enable LaTeX formatting
# Create custom templates
```

## Performance

- **Report generation time**: 2-5 minutes per report (depends on hardware)
- **Model size**: 4.1 GB (first download only)
- **Memory usage**: ~6-8 GB while generating

## Examples

### Running Full Pipeline with Reports

```powershell
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run pipeline (automatic)
cd "c:\Users\leela\Downloads\Telegram Desktop\plasma_reactor-main\plasma_reactor-main"
& "C:/Users/leela/Downloads/Telegram Desktop/plasma_reactor-main/.venv/Scripts/python.exe" run_complete_plasma_control.py

# Wait 3-5 minutes, then check for:
# - PLASMA_CONTROL_REPORT.md/html
# - PLASMA_CONTROL_TECHNICAL_REVIEW.md/html
```

### Manual Report Generation

```python
from ollama_report_generator import generate_reports

# Generate reports with custom data
control_data = {
    'final_reward': 250,
    'mean_eval_reward': 300,
    'eval_std': 0.5,
    'deployment_reward': 350,
    'initial_q95': 2.9,
    'final_q95': 2.2,
    'operational': True
}

reports = generate_reports(control_data)
print(reports)
```

## System Architecture

```
┌─────────────────────────────────────────┐
│   run_complete_plasma_control.py         │
│   (Main Pipeline + Report Integration)   │
└──────────────┬──────────────────────────┘
               │
      Step 5: Report Generation
               │
    ┌──────────▼─────────┐
    │  generate_llm_reports()
    └──────────┬──────────┘
               │
    ┌──────────▼────────────────────┐
    │  ollama_report_generator.py    │
    │  ├─ OllamaReportGenerator      │
    │  ├─ generate_plasma_report()   │
    │  ├─ generate_technical_review()│
    │  └─ save_md/html_report()      │
    └──────────┬────────────────────┘
               │
    ┌──────────▼────────────────────┐
    │   Ollama Service (localhost)    │
    │   Running llama3.2 Model        │
    └────────────────────────────────┘
```

## Integration Points

### In `run_complete_plasma_control.py`:

```python
# Import the report generator
from ollama_report_generator import generate_reports

# Call after deployment
def generate_llm_reports(training_summary):
    return generate_reports(training_summary)

# In main():
training_summary = {...}
generate_llm_reports(training_summary)
```

## Future Enhancements

- [ ] PDF report generation
- [ ] LaTeX/equation support
- [ ] Multi-language reports
- [ ] Customizable report templates
- [ ] Report comparison (before/after improvements)
- [ ] Automated performance recommendations
- [ ] Real-time report updates during training

## Support

If you encounter issues:

1. Verify Ollama is installed: `ollama --version`
2. Check service is running: `ollama serve`
3. Test connection: Check `http://localhost:11434` in browser
4. Review logs: Check terminal output from `ollama serve`
5. Manually pull model: `ollama pull llama3.2`

## References

- Ollama: https://ollama.ai
- Llama 3.2: https://llama.meta.com
- Plasma Control System: See PLASMA_CONTROL_SUCCESS_DOCUMENTATION.md

---

**Status**: ✅ Ready for use  
**Last Updated**: February 17, 2026
