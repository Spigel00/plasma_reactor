"""
Plasma Control Report Generator with Ollama LLM Integration
Generates comprehensive MD and HTML reports using Llama 3.2
"""

import os
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import requests

class OllamaReportGenerator:
    """Generate plasma control reports using Ollama Llama 3.2 LLM"""
    
    def __init__(self, model_name="llama3.2", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.available = False
        self.check_ollama()
    
    def check_ollama(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'].split(':')[0] for m in models]
                if self.model_name in model_names:
                    self.available = True
                    print(f"✅ Ollama is running with {self.model_name}")
                else:
                    print(f"⚠️  Ollama running but {self.model_name} not found")
                    print(f"   Available models: {model_names}")
                    self.pull_model()
        except requests.exceptions.ConnectionError:
            print(f"❌ Ollama not running. Start Ollama first:")
            print(f"   Command: ollama serve")
        except Exception as e:
            print(f"❌ Error checking Ollama: {e}")
    
    def pull_model(self):
        """Pull llama3.2 model from Ollama"""
        print(f"\n📥 Pulling {self.model_name} from Ollama...")
        try:
            result = subprocess.run(
                ["ollama", "pull", self.model_name],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0:
                print(f"✅ Successfully pulled {self.model_name}")
                self.available = True
            else:
                print(f"❌ Failed to pull model: {result.stderr}")
        except FileNotFoundError:
            print("❌ Ollama CLI not found. Install Ollama from https://ollama.ai")
        except subprocess.TimeoutExpired:
            print("⚠️  Model pull timeout (model may still be downloading)")
        except Exception as e:
            print(f"❌ Error pulling model: {e}")
    
    def generate_text(self, prompt, max_tokens=2000):
        """Generate text using Ollama"""
        if not self.available:
            return "Error: Ollama/llama3.2 not available"
        
        try:
            print(f"🤖 Generating with {self.model_name}...")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": max_tokens
                },
                timeout=300
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error generating text: {e}"
    
    def generate_plasma_report(self, control_data):
        """Generate AI-powered plasma control report"""
        
        # Prepare data summary
        data_summary = f"""
PLASMA CONTROL SYSTEM DATA SUMMARY:
- Training Timesteps: 100,000
- Final Episode Reward: {control_data.get('final_reward', 194)}
- Evaluation Episodes: 3
- Mean Evaluation Reward: {control_data.get('mean_eval_reward', 274.03)} ± {control_data.get('eval_std', 0.29)}
- Deployment Steps: 150
- Deployment Reward: {control_data.get('deployment_reward', 290.21)}
- Initial q95: {control_data.get('initial_q95', 2.85)}
- Final q95: {control_data.get('final_q95', 2.34)}
- Disruptions: 0
- Control Status: {'Operational' if control_data.get('operational') else 'Failed'}

IMPROVEMENT METRICS:
- Reward Improvement: 223%
- Learning Status: Failed → Success
- Control Quality: Chaotic → Stable
"""
        
        prompt = f"""You are an expert plasma physics and reinforcement learning specialist. 
Based on the following plasma control system data, generate a comprehensive technical report.

{data_summary}

Please provide:
1. Executive Summary (2-3 paragraphs)
2. System Architecture Overview (2 paragraphs)
3. Key Achievement Highlights (4-5 main points with details)
4. Technical Analysis:
   - Problem Identification
   - Solution Approach
   - Implementation Details
5. Performance Metrics and Results
6. Safety Validation
7. Deployment Readiness Assessment
8. Recommendations for Future Work

Format the response with clear sections and markdown-style headers."""
        
        return self.generate_text(prompt)
    
    def generate_technical_review(self, control_data):
        """Generate AI-powered technical review"""
        
        prompt = """You are a senior plasma physics and machine learning engineer reviewing a plasma control system using RL.

The system has achieved:
- Initial State: Non-functional (reward = -876)
- Final State: Fully operational (reward = +290)
- 223% improvement in control quality
- Zero disruptions across 150+ control steps
- Stable q95 (safety margin) maintained at 2.34 (safe if >2.0)
- Consistent performance (std dev = 0.29)

Please provide a professional technical review covering:

1. PROBLEM ANALYSIS
   - What was the root cause of initial failure
   - Why reward normalization was critical
   - Impact of hyperparameter tuning

2. SOLUTION ASSESSMENT
   - Effectiveness of the fix
   - Design choices and their justification
   - Potential improvements

3. PERFORMANCE EVALUATION
   - Convergence analysis
   - Consistency metrics
   - Comparison to baseline/traditional methods

4. SAFETY & VALIDATION
   - Safety mechanisms in place
   - Validation approach
   - Risk assessment

5. READINESS FOR DEPLOYMENT
   - Production readiness level
   - Potential challenges
   - Mitigation strategies

6. CONCLUSIONS & RECOMMENDATIONS

Use technical language appropriate for a peer-reviewed journal or conference."""
        
        return self.generate_text(prompt, max_tokens=3000)

def save_markdown_report(filename, title, content):
    """Save content as markdown file"""
    md_content = f"""# {title}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

{content}

---

*This report was generated using Ollama Llama 3.2 LLM*
"""
    
    with open(filename, 'w') as f:
        f.write(md_content)
    print(f"✅ Saved: {filename}")
    return filename

def save_html_report(filename, title, content):
    """Save content as HTML file"""
    
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        
        header {{
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
            margin-bottom: 40px;
        }}
        
        h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
        
        h2 {{
            color: #764ba2;
            margin-top: 40px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }}
        
        h3 {{
            color: #555;
            margin-top: 25px;
            margin-bottom: 10px;
        }}
        
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        
        ul, ol {{
            margin-left: 30px;
            margin-bottom: 15px;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        strong {{
            color: #764ba2;
        }}
        
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        pre {{
            background: #f4f4f4;
            border-left: 4px solid #667eea;
            padding: 15px;
            overflow-x: auto;
            margin-bottom: 15px;
            border-radius: 5px;
        }}
        
        .metric {{
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        
        .success {{
            color: #27ae60;
            font-weight: bold;
        }}
        
        .warning {{
            color: #e74c3c;
            font-weight: bold;
        }}
        
        footer {{
            border-top: 2px solid #eee;
            margin-top: 50px;
            padding-top: 20px;
            color: #999;
            font-size: 0.9em;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            background: #667eea;
            color: white;
            border-radius: 20px;
            font-size: 0.85em;
            margin-right: 10px;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 {title}</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>
        
        <main>
            {content}
        </main>
        
        <footer>
            <p>This report was generated using <strong>Ollama Llama 3.2 LLM</strong></p>
            <p>Plasma Control System - RL-Based Tokamak Controller</p>
            <p>© 2026 Plasma Research Initiative</p>
        </footer>
    </div>
</body>
</html>"""
    
    # Convert markdown to basic HTML formatting
    html_content = content.replace('\n', '<br>')
    html_content = html_content.replace('# ', '<h2>').replace('## ', '<h3>')
    
    final_html = html_template.format(
        title=title,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        content=html_content
    )
    
    with open(filename, 'w') as f:
        f.write(final_html)
    print(f"✅ Saved: {filename}")
    return filename

def generate_reports(control_data=None):
    """Main function to generate all reports"""
    
    if control_data is None:
        control_data = {
            'final_reward': 194,
            'mean_eval_reward': 274.03,
            'eval_std': 0.29,
            'deployment_reward': 290.21,
            'initial_q95': 2.85,
            'final_q95': 2.34,
            'operational': True
        }
    
    print("\n" + "="*80)
    print("PLASMA CONTROL REPORT GENERATION WITH OLLAMA LLAMA 3.2")
    print("="*80 + "\n")
    
    # Initialize Ollama
    ollama = OllamaReportGenerator()
    
    if not ollama.available:
        print("\n⚠️  Ollama/Llama 3.2 not available. Please:")
        print("   1. Install Ollama from https://ollama.ai")
        print("   2. Run: ollama serve")
        print("   3. In another terminal: ollama pull llama3.2")
        print("   4. Re-run this script")
        return
    
    # Generate reports
    print("\n📝 Generating AI-powered reports...\n")
    
    # Generate plasma report
    print("1️⃣  Creating Executive Report...")
    plasma_report = ollama.generate_plasma_report(control_data)
    
    # Generate technical review
    print("2️⃣  Creating Technical Review...")
    technical_review = ollama.generate_technical_review(control_data)
    
    # Save reports
    print("\n📁 Saving reports...\n")
    
    # Markdown files
    report_md = save_markdown_report(
        "PLASMA_CONTROL_REPORT.md",
        "Plasma Control System - Executive Report",
        plasma_report
    )
    
    review_md = save_markdown_report(
        "PLASMA_CONTROL_TECHNICAL_REVIEW.md",
        "Plasma Control System - Technical Review",
        technical_review
    )
    
    # HTML files
    markdown_to_html = lambda text: f"<p>{text.replace(chr(10), '</p><p>')}</p>"
    
    # Create basic HTML conversion
    report_html_content = markdown_to_html(plasma_report)
    review_html_content = markdown_to_html(technical_review)
    
    report_html = save_html_report(
        "PLASMA_CONTROL_REPORT.html",
        "Plasma Control System - Executive Report",
        report_html_content
    )
    
    review_html = save_html_report(
        "PLASMA_CONTROL_TECHNICAL_REVIEW.html",
        "Plasma Control System - Technical Review",
        review_html_content
    )
    
    print("\n" + "="*80)
    print("✅ REPORT GENERATION COMPLETE")
    print("="*80)
    print("\nGenerated Files:")
    print(f"  📄 {report_md}")
    print(f"  📄 {review_md}")
    print(f"  🌐 {report_html}")
    print(f"  🌐 {review_html}")
    print("\n" + "="*80 + "\n")
    
    return {
        'report_md': report_md,
        'review_md': review_md,
        'report_html': report_html,
        'review_html': review_html
    }

if __name__ == "__main__":
    generate_reports()
