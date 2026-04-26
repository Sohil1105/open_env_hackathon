"""
generate_charts.py
------------------
Generates a self-contained HTML dashboard (training_dashboard.html) with
interactive Chart.js plots — no matplotlib DLLs required.

Charts produced:
  1. Reward over Steps  (raw + rolling average)
  2. Loss over Steps    (raw + rolling average)
  3. Before vs After bar chart (reward comparison)
  4. Per-category reward breakdown (if decision/category field exists)

Run:  python generate_charts.py
"""

import os, sys, json, csv, statistics

# ── Config ────────────────────────────────────────────────────────────────────
TRAINING_LOG   = "training_log.csv"
BASELINE_FILE  = "baseline_outputs.json"
FINETUNED_FILE = "finetuned_outputs.json"
OUTPUT_HTML    = "training_dashboard.html"
ROLLING_WINDOW = 10

# ── Helpers ───────────────────────────────────────────────────────────────────
def rolling_avg(values, window=10):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(round(sum(values[start:i+1]) / (i - start + 1), 6))
    return result


def load_training_log(path):
    if not os.path.isfile(path):
        print(f"[WARN] Training log not found: {path}")
        return [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return [], [], [], []

    # Auto-detect column names
    cols = list(rows[0].keys())
    step_col     = next((c for c in cols if c.lower() in ["step","steps","epoch","global_step"]), None)
    reward_col   = next((c for c in cols if c.lower() in ["reward","mean_reward","avg_reward","score"]), None)
    # train loss: prefer "loss" or "training loss" over eval_loss
    train_loss_col = next((c for c in cols if c.lower() in ["loss","train_loss","training loss","training_loss"]), None)
    eval_loss_col  = next((c for c in cols if c.lower() in ["eval_loss","validation loss","val_loss","eval loss"]), None)

    print(f"[CSV] step={step_col}, reward={reward_col}, train_loss={train_loss_col}, eval_loss={eval_loss_col}")

    steps, rewards, train_losses, eval_losses = [], [], [], []
    for row in rows:
        try:
            steps.append(float(row[step_col]) if step_col else len(steps))
        except:
            steps.append(len(steps))
        try:
            rewards.append(float(row[reward_col]) if reward_col else None)
        except:
            rewards.append(None)
        try:
            train_losses.append(float(row[train_loss_col]) if train_loss_col else None)
        except:
            train_losses.append(None)
        try:
            eval_losses.append(float(row[eval_loss_col]) if eval_loss_col else None)
        except:
            eval_losses.append(None)

    return steps, rewards, train_losses, eval_losses


def load_json_outputs(path, label):
    if not os.path.isfile(path):
        print(f"[WARN] {label} file not found: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[OK] Loaded {label}: {len(data)} records")
    return data


def get_reward_scores(records):
    keys = ["reward_score","reward","score","mean_reward"]
    key = next((k for k in keys if any(k in r for r in records)), None)
    if not key:
        return []
    return [float(r[key]) for r in records if key in r and r[key] is not None]


def get_category_means(records):
    cat_keys = ["decision","risk_level","category","label","class"]
    cat_key = next((k for k in cat_keys if records and k in records[0]), None)
    if not cat_key:
        return {}
    rew_key = next((k for k in ["reward_score","reward","score"] if records and k in records[0]), None)
    if not rew_key:
        return {}
    buckets = {}
    for r in records:
        cat = str(r.get(cat_key,"?"))
        val = r.get(rew_key)
        if val is not None:
            buckets.setdefault(cat, []).append(float(val))
    return {cat: round(sum(v)/len(v), 4) for cat, v in buckets.items()}


# ── Load all data ─────────────────────────────────────────────────────────────
steps, rewards, train_losses, eval_losses = load_training_log(TRAINING_LOG)

reward_smooth      = rolling_avg([r for r in rewards      if r is not None], ROLLING_WINDOW) if any(r is not None for r in rewards)      else []
train_loss_smooth  = rolling_avg([l for l in train_losses if l is not None], ROLLING_WINDOW) if any(l is not None for l in train_losses) else []
eval_loss_smooth   = rolling_avg([l for l in eval_losses  if l is not None], ROLLING_WINDOW) if any(l is not None for l in eval_losses)  else []

# Pair steps with non-None values
rew_steps        = [steps[i] for i,r in enumerate(rewards)      if r is not None]
raw_rew          = [r for r in rewards      if r is not None]
train_loss_steps = [steps[i] for i,l in enumerate(train_losses) if l is not None]
raw_train_loss   = [l for l in train_losses if l is not None]
eval_loss_steps  = [steps[i] for i,l in enumerate(eval_losses)  if l is not None]
raw_eval_loss    = [l for l in eval_losses  if l is not None]

# Unified loss x-axis (all steps that have either loss)
loss_steps = sorted(set(train_loss_steps + eval_loss_steps))

before_records  = load_json_outputs(BASELINE_FILE,  "baseline")
after_records   = load_json_outputs(FINETUNED_FILE, "fine-tuned")

before_scores = get_reward_scores(before_records) if before_records else []
after_scores  = get_reward_scores(after_records)  if after_records  else []

mean_before = round(statistics.mean(before_scores), 4) if before_scores else 0
mean_after  = round(statistics.mean(after_scores),  4) if after_scores  else 0
std_before  = round(statistics.stdev(before_scores), 4) if len(before_scores) > 1 else 0
std_after   = round(statistics.stdev(after_scores),  4) if len(after_scores)  > 1 else 0
improvement = round(((mean_after - mean_before) / abs(mean_before)) * 100, 2) if mean_before != 0 else 0

print(f"\n{'='*50}")
print(f"  Mean reward BEFORE : {mean_before}  (±{std_before})")
print(f"  Mean reward AFTER  : {mean_after}  (±{std_after})")
print(f"  Improvement        : {improvement:+.2f}%")
print(f"{'='*50}\n")

before_cats = get_category_means(before_records) if before_records else {}
after_cats  = get_category_means(after_records)  if after_records  else {}
all_cats    = sorted(set(list(before_cats.keys()) + list(after_cats.keys())))

# ── Build HTML ────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Loan Underwriting RL Model — Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0f0f1a;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    padding: 24px;
  }}
  h1 {{
    text-align: center;
    font-size: 1.7rem;
    color: #ffffff;
    margin-bottom: 6px;
    letter-spacing: 0.5px;
  }}
  .subtitle {{
    text-align: center;
    color: #888;
    font-size: 0.9rem;
    margin-bottom: 32px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    max-width: 1400px;
    margin: 0 auto;
  }}
  .card {{
    background: #16213e;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  }}
  .card h2 {{
    font-size: 1rem;
    color: #aaa;
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .stat-row {{
    display: flex;
    justify-content: space-around;
    margin-bottom: 28px;
    flex-wrap: wrap;
    gap: 12px;
  }}
  .stat {{
    text-align: center;
    background: #16213e;
    border-radius: 10px;
    padding: 16px 28px;
    min-width: 150px;
  }}
  .stat .val {{
    font-size: 2rem;
    font-weight: 700;
    color: #4ecdc4;
  }}
  .stat .val.red {{ color: #ff6b6b; }}
  .stat .val.gold {{ color: #ffd93d; }}
  .stat .lbl {{ font-size: 0.78rem; color: #888; margin-top: 4px; }}
  canvas {{ max-height: 300px; }}
  @media (max-width: 900px) {{
    .grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<h1>Loan Underwriting RL Model &mdash; Training Dashboard</h1>
<p class="subtitle">Fine-tuning on Llama-3-8B &bull; Unsloth + TRL SFTTrainer &bull; Loan Risk Classification</p>

<div class="stat-row">
  <div class="stat">
    <div class="val red">{mean_before}</div>
    <div class="lbl">Mean Reward (Before)</div>
  </div>
  <div class="stat">
    <div class="val">{mean_after}</div>
    <div class="lbl">Mean Reward (After)</div>
  </div>
  <div class="stat">
    <div class="val gold">{'+' if improvement >= 0 else ''}{improvement}%</div>
    <div class="lbl">Reward Improvement</div>
  </div>
  <div class="stat">
    <div class="val" style="color:#c77dff">{len(train_loss_steps)}</div>
    <div class="lbl">Training Steps Logged</div>
  </div>
  <div class="stat">
    <div class="val" style="color:#ffd369">{round(raw_train_loss[-1], 4) if raw_train_loss else 'N/A'}</div>
    <div class="lbl">Final Train Loss</div>
  </div>
  <div class="stat">
    <div class="val" style="color:#c3f584">{round(raw_eval_loss[-1], 4) if raw_eval_loss else 'N/A'}</div>
    <div class="lbl">Final Val Loss</div>
  </div>
</div>

<div class="grid">

  <!-- Reward curve -->
  <div class="card">
    <h2>Reward over Training Steps</h2>
    <canvas id="rewardChart"></canvas>
  </div>

  <!-- Loss curve -->
  <div class="card">
    <h2>Loss over Training Steps</h2>
    <canvas id="lossChart"></canvas>
  </div>

  <!-- Before vs After -->
  <div class="card">
    <h2>Before vs After Fine-Tuning</h2>
    <canvas id="beforeAfterChart"></canvas>
  </div>

  <!-- Per-category -->
  <div class="card">
    <h2>Per-Category Reward Breakdown</h2>
    <canvas id="categoryChart"></canvas>
  </div>

</div>

<script>
const GRID_COLOR = 'rgba(255,255,255,0.08)';
const TICK_COLOR = '#888';

const baseOpts = {{
  responsive: true,
  plugins: {{
    legend: {{ labels: {{ color: '#ccc', boxWidth: 14, font: {{ size: 12 }} }} }},
    tooltip: {{ mode: 'index', intersect: false }}
  }},
  scales: {{
    x: {{ ticks: {{ color: TICK_COLOR, maxTicksLimit: 12 }}, grid: {{ color: GRID_COLOR }} }},
    y: {{ ticks: {{ color: TICK_COLOR }}, grid: {{ color: GRID_COLOR }} }}
  }}
}};

// ── 1. Reward chart ──────────────────────────────────────────────────────────
new Chart(document.getElementById('rewardChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(rew_steps)},
    datasets: [
      {{
        label: 'Raw Reward',
        data: {json.dumps(raw_rew)},
        borderColor: 'rgba(78,205,196,0.3)',
        backgroundColor: 'transparent',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.3
      }},
      {{
        label: 'Rolling Avg (w={ROLLING_WINDOW})',
        data: {json.dumps(reward_smooth)},
        borderColor: '#4ecdc4',
        backgroundColor: 'rgba(78,205,196,0.1)',
        borderWidth: 2.5,
        pointRadius: 0,
        fill: true,
        tension: 0.4
      }}
    ]
  }},
  options: JSON.parse(JSON.stringify(baseOpts))
}});

// ── 2. Loss chart (Train + Validation) ───────────────────────────────────────
new Chart(document.getElementById('lossChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(train_loss_steps if train_loss_steps else eval_loss_steps)},
    datasets: [
      {{
        label: 'Train Loss (raw)',
        data: {json.dumps(raw_train_loss)},
        borderColor: 'rgba(255,107,107,0.3)',
        backgroundColor: 'transparent',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.3
      }},
      {{
        label: 'Train Loss (avg w={ROLLING_WINDOW})',
        data: {json.dumps(train_loss_smooth)},
        borderColor: '#ff6b6b',
        backgroundColor: 'rgba(255,107,107,0.08)',
        borderWidth: 2.5,
        pointRadius: 0,
        fill: false,
        tension: 0.4
      }},
      {{
        label: 'Val Loss (raw)',
        data: {json.dumps(raw_eval_loss)},
        borderColor: 'rgba(255,211,105,0.35)',
        backgroundColor: 'transparent',
        borderWidth: 1,
        pointRadius: 3,
        pointBackgroundColor: 'rgba(255,211,105,0.6)',
        tension: 0.3
      }},
      {{
        label: 'Val Loss (avg w={ROLLING_WINDOW})',
        data: {json.dumps(eval_loss_smooth)},
        borderColor: '#ffd369',
        backgroundColor: 'rgba(255,211,105,0.08)',
        borderWidth: 2.5,
        pointRadius: 0,
        fill: false,
        tension: 0.4
      }}
    ]
  }},
  options: JSON.parse(JSON.stringify(baseOpts))
}});

// ── 3. Before vs After bar chart ─────────────────────────────────────────────
new Chart(document.getElementById('beforeAfterChart'), {{
  type: 'bar',
  data: {{
    labels: ['Before (Baseline)', 'After (Fine-tuned)'],
    datasets: [{{
      label: 'Mean Reward Score',
      data: [{mean_before}, {mean_after}],
      backgroundColor: ['rgba(255,107,107,0.75)', 'rgba(78,205,196,0.75)'],
      borderColor:     ['#ff6b6b', '#4ecdc4'],
      borderWidth: 2,
      borderRadius: 6
    }}]
  }},
  options: {{
    ...JSON.parse(JSON.stringify(baseOpts)),
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{
        afterLabel: ctx => ctx.dataIndex === 1
          ? `Improvement: {'+' if improvement >= 0 else ''}{improvement}%`
          : ''
      }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#ccc', font: {{ size: 13 }} }}, grid: {{ color: GRID_COLOR }} }},
      y: {{
        ticks: {{ color: TICK_COLOR }},
        grid: {{ color: GRID_COLOR }},
        min: 0,
        max: 1.0
      }}
    }}
  }}
}});

// ── 4. Per-category breakdown ────────────────────────────────────────────────
const cats = {json.dumps(all_cats)};
const catBefore = cats.map(c => ({{
  {', '.join(f'"{k}": {v}' for k, v in before_cats.items())}
}})[c] || 0);
const catAfter = cats.map(c => ({{
  {', '.join(f'"{k}": {v}' for k, v in after_cats.items())}
}})[c] || 0);

new Chart(document.getElementById('categoryChart'), {{
  type: 'bar',
  data: {{
    labels: cats,
    datasets: [
      {{
        label: 'Before',
        data: catBefore,
        backgroundColor: 'rgba(255,107,107,0.7)',
        borderColor: '#ff6b6b',
        borderWidth: 2,
        borderRadius: 5
      }},
      {{
        label: 'After',
        data: catAfter,
        backgroundColor: 'rgba(78,205,196,0.7)',
        borderColor: '#4ecdc4',
        borderWidth: 2,
        borderRadius: 5
      }}
    ]
  }},
  options: {{
    ...JSON.parse(JSON.stringify(baseOpts)),
    scales: {{
      x: {{ ticks: {{ color: '#ccc' }}, grid: {{ color: GRID_COLOR }} }},
      y: {{ ticks: {{ color: TICK_COLOR }}, grid: {{ color: GRID_COLOR }}, min: 0, max: 1.0 }}
    }}
  }}
}});
</script>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"[DONE] Dashboard saved -> {OUTPUT_HTML}")
print(f"       Open it in your browser: start {OUTPUT_HTML}")
