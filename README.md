---
title: Loan Underwriting OpenEnv
emoji: "\U0001F3E6"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Loan Underwriting & Risk Assessment — OpenEnv

An **OpenEnv-compliant** reinforcement learning environment that simulates a bank's loan underwriting desk. AI agents evaluate applicant financial profiles and make multi-component lending decisions with partial credit scoring.

Built for the **Scaler x Meta PyTorch Hackathon (OpenEnv Round 1)**.

---

## Environment Overview

At each episode, the agent receives a **loan applicant profile** containing:
- Income, credit score, existing debt
- Employment type and tenure
- Loan amount requested and repayment tenure
- Monthly expenses, collateral status, default history

The agent must make **three decisions**:

| Decision | Options | Weight |
|----------|---------|--------|
| **Risk Level** | Low / Medium / High | 40% |
| **Loan Decision** | Approve / Conditional Approve / Reject | 35% |
| **Interest Rate Tier** | 7-9% / 10-13% / 14%+ | 25% |

Scoring uses **partial credit** — adjacent classifications earn partial points, and **logical consistency** across all three decisions adds a bonus/penalty modifier.

---

## API Endpoints

### Health Check
```bash
# Must return HTTP 200
curl https://Sourav0511-open-env-hackathon.hf.space/

# Detailed health check — returns {"status": "ok", ...}
curl https://Sourav0511-open-env-hackathon.hf.space/health
```

### Reset Environment
```bash
# Reset with default (easy) task
curl -X POST https://Sourav0511-open-env-hackathon.hf.space/reset

# Reset with specific task
curl -X POST https://Sourav0511-open-env-hackathon.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_salaried_high_credit"}'
```

Available task IDs:
- `easy_salaried_high_credit` — Salaried, 785 credit score
- `medium_self_employed_moderate` — Self-employed, 665 credit score
- `hard_freelancer_complex` — Freelancer, 572 credit score

### Submit Decision (Step)
```bash
curl -X POST https://Sourav0511-open-env-hackathon.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "risk_level": "Low",
    "loan_decision": "Approve",
    "interest_rate_tier": "7-9%",
    "reasoning": "Excellent profile with high credit score."
  }'
```

Returns: `{"state": {...}, "reward": 0.95, "done": true, "info": {...}}`

### Get Current State
```bash
curl https://Sourav0511-open-env-hackathon.hf.space/state
```

### List Tasks
```bash
curl https://Sourav0511-open-env-hackathon.hf.space/tasks
```

### Get OpenEnv Spec
```bash
curl https://Sourav0511-open-env-hackathon.hf.space/openenv.yaml
```

---

## Tasks (Progressive Difficulty)

### Task 1: Easy — Salaried High-Credit Profile
- **Profile**: Salaried employee, credit score 785, low DTI ratio, collateral available
- **Focus**: Primarily tests risk level classification
- **Expected**: Low risk -> Approve -> 7-9%

### Task 2: Medium — Self-Employed Moderate Profile
- **Profile**: Self-employed, credit score 665, moderate debt, 1 previous default
- **Focus**: Tests risk classification AND loan decision balancing
- **Expected**: Medium risk -> Conditional Approve -> 10-13%

### Task 3: Hard — Freelancer Complex Profile
- **Profile**: Freelancer, credit score 572, high debt, no collateral, 2 defaults
- **Focus**: All three decisions must align logically
- **Expected**: High risk -> Reject -> 14%+

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `applicant_name` | string | Full name of the applicant |
| `age` | integer | Age in years |
| `annual_income` | float | Annual income in USD |
| `credit_score` | integer | FICO score (300-850) |
| `existing_debt` | float | Total existing debt in USD |
| `employment_type` | enum | salaried / self_employed / freelancer / contract / unemployed |
| `employment_years` | float | Years in current role |
| `loan_amount_requested` | float | Requested loan amount in USD |
| `repayment_tenure_months` | integer | Repayment period in months |
| `monthly_expenses` | float | Average monthly expenses in USD |
| `has_collateral` | boolean | Whether collateral is offered |
| `previous_defaults` | integer | Number of previous loan defaults |
| `task_description` | string | What the agent must decide |
| `task_id` | string | Unique task identifier |
| `task_difficulty` | string | easy / medium / hard |

## Action Space

| Field | Type | Options |
|-------|------|---------|
| `risk_level` | enum | Low / Medium / High |
| `loan_decision` | enum | Approve / Conditional Approve / Reject |
| `interest_rate_tier` | enum | 7-9% / 10-13% / 14%+ |
| `reasoning` | string | Brief explanation (optional) |

---

## Scoring Details

### Component Weights
- **Risk Level**: 0.40 (40%)
- **Loan Decision**: 0.35 (35%)
- **Interest Rate Tier**: 0.25 (25%)

### Partial Credit
- **Exact match**: 1.0
- **Adjacent category** (off by one): 0.30-0.35
- **Completely wrong** (off by two): 0.0

### Consistency Bonus
- Logically aligned decisions: **+0.05 to +0.10**
- Contradictory decisions: **-0.05 to -0.10**
- Final score clamped to **[0.0, 1.0]**

---

## Project Structure

```
loan-underwriting-openenv/
├── environment/
│   ├── __init__.py      # Package exports
│   ├── env.py           # Main OpenEnv environment class
│   ├── tasks.py         # 3 task definitions (easy/medium/hard)
│   ├── graders.py       # Automated graders returning 0.0-1.0
│   ├── models.py        # Pydantic typed Observation, Action, State
│   └── rewards.py       # Partial reward function logic
├── app.py               # FastAPI server for HF Space deployment
├── inference.py          # Baseline LLM agent script
├── openenv.yaml          # OpenEnv spec metadata
├── Dockerfile            # Container for HF Spaces (port 7860)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Environment Variables

These are configured as **Secrets** in your HF Space settings:

| Variable | Description | Example |
|----------|-------------|---------|
| `API_BASE_URL` | OpenAI-compatible API endpoint | `https://api-inference.huggingface.co/v1` |
| `MODEL_NAME` | Model to use for inference | `meta-llama/Llama-3.1-8B-Instruct` |
| `HF_TOKEN` | Hugging Face API token | `hf_xxxxxxxxxxxx` |

> **Important**: Never hardcode secrets. Always use HF Space Secrets (Settings > Variables and Secrets).

---

## Run Locally

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Start the Server
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run with Docker
```bash
docker build -t loan-underwriting-openenv .
docker run -p 7860:7860 loan-underwriting-openenv
```

### Run Inference
```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_hf_token_here"

python inference.py
```

---

## Source Code

GitHub: [https://github.com/Sohil1105/open_env_hackathon](https://github.com/Sohil1105/open_env_hackathon)

## License

MIT License

---

### Task 4: Bankruptcy Recovery (Edge Case)
- **Profile**: Credit rebuilt to 680, bankrupt 7 years ago, low debt
- **Focus**: Evaluates ability to handle historical negative marks
- **Expected**: Medium risk -> Conditional Approve -> 10-13%

### Task 5: Joint Applicants (Edge Case)
- **Profile**: Combined income $120k, primary credit 720
- **Focus**: Evaluates handling of combined application metrics
- **Expected**: Low risk -> Approve -> 7-9%

### Example curl commands for Tasks 4 & 5
```bash
curl -X POST https://Sourav0511-open-env-hackathon.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "bankruptcy_recovery_edge1"}'

curl -X POST https://Sourav0511-open-env-hackathon.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "joint_applicants_edge2"}'
```

