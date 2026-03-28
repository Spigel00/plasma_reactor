#!/usr/bin/env bash
set -euo pipefail

# Run plasma deployment, archive generated plot, and create a run-specific HTML report.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
TS="$(date +%Y%m%d_%H%M%S)"

PLOTS_DIR="outputs/plots"
REPORTS_DIR="outputs/reports"
LOGS_DIR="outputs/logs"

mkdir -p "$PLOTS_DIR" "$REPORTS_DIR" "$LOGS_DIR"

LOG_FILE="$LOGS_DIR/plasma_run_${TS}.log"
PLOT_FILE="$PLOTS_DIR/plasma_control_results_${TS}.png"
REPORT_FILE="$REPORTS_DIR/plasma_run_${TS}.html"

echo "Running deployment with: $PYTHON_BIN plasma_deployment.py"
"$PYTHON_BIN" plasma_deployment.py | tee "$LOG_FILE"

if [[ ! -f "plasma_control_results.png" ]]; then
  echo "Error: plasma_control_results.png was not generated."
  exit 1
fi

cp "plasma_control_results.png" "$PLOT_FILE"

cat > "$REPORT_FILE" <<EOF
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Plasma Run Report ${TS}</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --card: #ffffff;
      --ink: #1f2937;
      --muted: #4b5563;
      --accent: #0f766e;
      --border: #dbe2ea;
    }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #edf3ff 0%, #f7fafc 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 1200px;
      margin: 32px auto;
      padding: 0 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
      margin-bottom: 16px;
    }
    h1 {
      margin: 0 0 8px;
      color: var(--accent);
      font-size: 1.8rem;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 1.2rem;
    }
    p {
      color: var(--muted);
      margin: 6px 0;
    }
    .meta {
      display: grid;
      gap: 6px;
    }
    img {
      width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }
    a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Plasma Model Run Report</h1>
      <div class="meta">
        <p><strong>Timestamp:</strong> ${TS}</p>
        <p><strong>Runner:</strong> ${PYTHON_BIN} plasma_deployment.py</p>
        <p><strong>Log file:</strong> <a href="../logs/plasma_run_${TS}.log">plasma_run_${TS}.log</a></p>
        <p><strong>Plot file:</strong> <a href="../plots/plasma_control_results_${TS}.png">plasma_control_results_${TS}.png</a></p>
      </div>
    </div>
    <div class="card">
      <h2>Control Performance Plot</h2>
      <img src="../plots/plasma_control_results_${TS}.png" alt="Plasma control results" />
    </div>
  </div>
</body>
</html>
EOF

echo ""
echo "Run complete. Artifacts:"
echo "  Plot:   $PLOT_FILE"
echo "  Report: $REPORT_FILE"
echo "  Log:    $LOG_FILE"
