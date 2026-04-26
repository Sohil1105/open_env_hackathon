# 🏦 Your Loan Underwriting OpenEnv — Complete Deep Dive

> A friendly, simple explanation of everything you built — and where to take it next.

---

## 1. 📝 Simple Project Summary (5 Sentences)

1. **You built a virtual bank desk** where an AI agent acts as a loan officer deciding whether to approve or reject someone's loan application.
2. **The agent reads a person's financial profile** — their income, credit score, debts, job type — just like a real banker would read a file.
3. **The agent then makes 3 decisions**: How risky is this person? Should we approve, conditionally approve, or reject? What interest rate should we charge?
4. **A grading system automatically scores** those decisions by comparing them to the "correct" answers, giving partial credit if the agent is close but not perfect.
5. **The whole thing runs as a web service** that can be hosted on Hugging Face, so anyone can test their own AI agent against your bank scenarios.

> **Think of it like this**: You built a loan-officer training simulator with a built-in answer key and automatic grading — and you let AI students take the test.

---

## 2. 🔄 How Your Environment Works (Step by Step)

### What happens when it STARTS (`reset`)

```
1. Someone says "Start task #1"
2. The environment pulls out a pre-made applicant file (e.g., "Rajesh Kumar Sharma")
3. It packages that file into an "observation" — all the numbers the agent needs to see
4. It clears all previous scores and marks the episode as "in progress"
5. It returns the observation to the agent: "Here's your applicant. What do you decide?"
```

### What happens at EACH STEP (`step`)

```
1. The agent sends back its answer: { risk: "Low", decision: "Approve", rate: "7-9%" }
2. The environment checks this answer against the correct answer stored in "ground truth"
3. It grades each of the 3 decisions separately (partial credit allowed!)
4. It checks if the 3 decisions are logically consistent with each other
5. It calculates a final reward score between 0.01 and 0.99
6. It marks the episode as "done" (it's a single-step environment)
7. It returns: updated state, reward, done=True, and detailed feedback
```

### What the agent SEES (observation)

| Field | Example | What it means |
|-------|---------|---------------|
| `applicant_name` | "Rajesh Kumar Sharma" | Who is applying |
| `age` | 35 | How old they are |
| `annual_income` | $1,200,000 | How much they earn per year |
| `credit_score` | 785 | Their creditworthiness (300-850) |
| `existing_debt` | $150,000 | How much they already owe |
| `employment_type` | "salaried" | Job stability indicator |
| `employment_years` | 8.5 | How long at current job |
| `loan_amount_requested` | $500,000 | How much they want to borrow |
| `repayment_tenure_months` | 60 | How long to pay it back |
| `monthly_expenses` | $45,000 | Monthly spending |
| `has_collateral` | True | Do they have property as backup? |
| `previous_defaults` | 0 | Have they failed to pay loans before? |
| `task_description` | "Evaluate this..." | Instructions for the agent |

### What the agent DOES (action)

The agent must output exactly 3 decisions:

| Decision | Options | What it means |
|----------|---------|---------------|
| **Risk Level** | Low / Medium / High | How dangerous is this borrower? |
| **Loan Decision** | Approve / Conditional Approve / Reject | Should we give them money? |
| **Interest Rate Tier** | 7-9% / 10-13% / 14%+ | If yes, how much extra do they pay? |

### How it gets SCORED (reward)

```
Final Score = (Risk Score × 0.40) + (Decision Score × 0.35) + (Rate Score × 0.25) + Consistency Bonus

Example: Agent gets all 3 correct on the easy task:
  = (0.99 × 0.40) + (0.99 × 0.35) + (0.99 × 0.25) + 0.10
  = 0.396 + 0.3465 + 0.2475 + 0.10
  = 0.99 (clamped to max 0.99)
```

---

## 3. 📋 Your 5 Tasks Explained Simply

### Task 1: Easy — Rajesh Kumar Sharma (Salaried, High Credit)
**What it is**: A "slam dunk" application. Rajesh has an excellent credit score (785), a stable salaried job for 8.5 years, earns $1.2M/year, and only owes $150K. He wants a $500K loan with collateral and zero defaults.
**Why it's easy**: Every single indicator points to "approve." There's no ambiguity. If the agent can't get this right, it can't do anything.

### Task 2: Medium — Priya Venkatesh (Self-Employed, Average Credit)
**What it is**: A "mixed signals" application. Priya is self-employed with average credit (665), moderate debt, one previous default, BUT she has collateral and 5 years of business history.
**Why it's harder**: The agent must weigh competing factors — some good, some bad — and land on "Conditional Approve" instead of a clear yes or no.

### Task 3: Hard — Arjun Mehta (Freelancer, Low Credit)
**What it is**: A "red flags everywhere" application. Arjun is a freelancer with poor credit (572), massive debt (90% of income!), wants to borrow 155% of his income, no collateral, and 2 previous defaults.
**Why it's hardest**: The agent must recognize that even though the numbers are extreme, it needs to correctly align all 3 decisions: High risk → Reject → 14%+ rate. All three must be logically consistent.

### Task 4: Medium — John Doe (Bankruptcy Recovery)
**What it is**: An "edge case" — someone who went bankrupt 7 years ago but has rebuilt their life. Credit is now 680, debt is low, but the loan is big ($120K) with no collateral.
**Why it's tricky**: The agent needs to recognize recovery after bankruptcy — don't punish someone forever, but don't fully trust them either. This tests nuanced judgment.

### Task 5: Easy — The Smiths (Joint Applicants)
**What it is**: A married couple applying together with combined income of $120K, good credit (720), low debt, collateral, and a $300K loan over 30 years.
**Why it's different**: Tests whether the agent handles "joint applicants" — a slightly unusual format. The numbers are strong, so the answer is approve, but the agent needs to reason about combined profiles.

### Difficulty Progression Summary

```
Task 1 (Easy)   → One clear signal → correct answer obvious
Task 5 (Easy)   → Clear signal but unusual format (joint)
Task 2 (Medium) → Mixed signals → must weigh trade-offs
Task 4 (Medium) → Edge case → must understand context (bankruptcy recovery)
Task 3 (Hard)   → All bad signals → must get all 3 decisions consistent
```

---

## 4. 📊 How Your Grader Works

### The Grading Pipeline

```
Agent's Answer ──→ Grade Risk Level ──→ Score (0.01 - 0.99)
                ──→ Grade Loan Decision ──→ Score (0.01 - 0.99)
                ──→ Grade Interest Rate ──→ Score (0.01 - 0.99)
                ──→ Grade Consistency ──→ Bonus/Penalty (-0.10 to +0.10)
                                            ↓
                                    Weighted Total Score
```

### What is Semantic Similarity Scoring?

Instead of "right or wrong," your grader asks: **"How close was the agent?"**

**Example — Risk Level:**
| Agent Said | Correct Answer | Score | Why? |
|-----------|---------------|-------|------|
| Low | Low | 0.99 | Perfect match! |
| Low | Medium | 0.30 | One step off, partial credit |
| Low | High | 0.01 | Completely wrong, almost zero |

It's like a teacher giving a grade:
- 🅰️ "You said Low, answer is Low" → Full marks
- 🅱️ "You said Low, answer is Medium" → Partial credit (you were close!)
- 🅵 "You said Low, answer is High" → Almost no credit (you were way off!)

### Why is Partial Credit Important?

> **Without partial credit**: An agent that says "Medium" when the answer is "High" gets **0 points** — same as an agent that randomly says "Low". That's unfair!

> **With partial credit**: The "Medium" agent gets **0.30 out of 0.99** — it's acknowledged for being close. This helps the agent **learn faster** because it can see "I'm getting warmer."

Your grader uses three similarity maps (defined in [graders.py](file:///d:/Open_env_hackathon/loan-underwriting-openenv/environment/graders.py#L26-L60)):
- `RISK_SIMILARITY` — scores for risk level comparisons
- `DECISION_SIMILARITY` — scores for loan decisions
- `RATE_SIMILARITY` — scores for interest rate tiers

---

## 5. 💰 Your Reward Function Explained

### The Formula

```
Reward = (Risk Score × 0.40) + (Decision Score × 0.35) + (Rate Score × 0.25) + Consistency Bonus
```

- **Risk Level** is weighted highest (40%) because it's the foundation — get this wrong and everything else falls apart
- **Loan Decision** is next (35%) because it's the actual business outcome
- **Interest Rate** is lowest (25%) because it follows from the other two

### What is the Consistency Bonus?

The consistency bonus/penalty checks: **"Do your 3 decisions make logical sense together?"**

**Good combinations (bonus +0.10):**
- Low risk + Approve + 7-9% → ✅ Makes perfect sense!
- Medium risk + Conditional Approve + 10-13% → ✅ Logical!
- High risk + Reject + 14%+ → ✅ Consistent!

**Bad combinations (penalty −0.10):**
- Low risk + Reject → ❌ Why would you reject a safe person?
- High risk + Approve → ❌ Why would you approve a dangerous borrower?
- High risk + 7-9% → ❌ Lowest rate for the riskiest person?

### Simple Example

**Scenario**: Agent evaluates Rajesh (the easy task).

| Decision | Agent Says | Correct | Score |
|----------|-----------|---------|-------|
| Risk Level | Low ✅ | Low | 0.99 |
| Loan Decision | Approve ✅ | Approve | 0.99 |
| Interest Rate | 7-9% ✅ | 7-9% | 0.99 |

```
Base = (0.99 × 0.40) + (0.99 × 0.35) + (0.99 × 0.25) = 0.99
Consistency Bonus = +0.10 (Low + Approve + 7-9% = perfect match)
Raw Total = 0.99 + 0.10 = 1.09
Clamped = 0.99 (maximum allowed)

Final Reward: 0.99 🎉
```

**What if the agent messes up one thing?**

| Decision | Agent Says | Correct | Score |
|----------|-----------|---------|-------|
| Risk Level | Low ✅ | Low | 0.99 |
| Loan Decision | Conditional Approve ⚠️ | Approve | 0.35 |
| Interest Rate | 7-9% ✅ | 7-9% | 0.99 |

```
Base = (0.99 × 0.40) + (0.35 × 0.35) + (0.99 × 0.25) = 0.396 + 0.1225 + 0.2475 = 0.766
Consistency Bonus = 0.0 (not perfectly consistent, not contradictory)
Final Reward: 0.766 — decent but not great
```

---

## 6. 🛠️ Your Tech Stack

| Technology | What it does | Why you needed it |
|-----------|-------------|-------------------|
| **Python 3.10** | The programming language everything is written in | Required by the hackathon spec |
| **Pydantic v2** | Defines and validates all your data structures (profiles, actions, scores) | Ensures agents can't submit garbage data; auto-validates types |
| **FastAPI** | Creates the web server with REST API endpoints (`/reset`, `/step`, `/grade`) | Agents communicate with your environment over HTTP |
| **Uvicorn** | Runs the FastAPI server (ASGI server) | The engine that makes FastAPI actually listen for web requests |
| **OpenAI Python SDK** | Sends applicant profiles to an LLM and gets loan decisions | Your baseline agent uses an LLM to make decisions |
| **PyYAML** | Loads/serves the `openenv.yaml` config file | OpenEnv spec requires a YAML definition |
| **openenv-core** | The official OpenEnv framework package | Required by the hackathon — makes your env "OpenEnv-compliant" |
| **Docker** | Packages everything into a container for deployment | Needed for Hugging Face Spaces deployment |
| **Hugging Face Spaces** | Hosts your environment online so judges/agents can access it | The hackathon's required deployment platform |
| **Meta LLaMA 3.2 3B** | The default LLM model your baseline agent uses | Free model on HF Inference API for the baseline |

---

## 7. 💪 Project Strengths

### What you did well

| Strength | Why it matters |
|----------|---------------|
| **Real-world domain** | Loan underwriting is a genuine, important problem — not a toy example |
| **Partial credit scoring** | Much more sophisticated than binary right/wrong — shows you understand RL reward shaping |
| **Consistency checking** | You don't just grade individual answers — you check if they make sense *together* |
| **5 progressive tasks** | From easy to hard with edge cases — demonstrates curriculum design thinking |
| **Edge cases** (bankruptcy recovery, joint applicants) | Shows depth — you didn't stop at 3 obvious tasks |
| **Robust input parsing** | Your Action validators handle "approve", "Approve", "APPROVE", "deny" → all work |
| **Heuristic ground truth generator** | Custom profiles get auto-graded — not just the predefined 5 tasks |
| **Clean code architecture** | Separate files for models, tasks, graders, rewards, env, server — very professional |
| **Comprehensive API** | `/reset`, `/step`, `/state`, `/grade`, `/tasks`, `/health`, `/openenv.yaml` — all endpoints covered |
| **Graceful error handling** | API errors return default "medium" actions instead of crashing |

### What makes it unique

1. **Multi-component grading** — Most hackathon envs grade one thing. You grade three things with weighted scores.
2. **Semantic similarity** — You built a mini "distance metric" between categorical answers.
3. **Consistency as a reward signal** — This is a real RL research concept (reward shaping for coherence).
4. **Production-ready deployment** — Docker + HF Spaces + health checks + CORS = actually deployable.

### What would impress judges

- The **weighted multi-component scoring** mimics how real banks evaluate loan officers.
- The **consistency bonus/penalty** is a genuinely clever reward engineering idea.
- The **edge case tasks** (bankruptcy recovery, joint applicants) show domain expertise.
- The **clean separation** of `models → tasks → graders → rewards → env → server` is textbook software engineering.

---

## 8. ⚠️ Project Weaknesses

### What is missing

| Gap | Why it matters |
|-----|---------------|
| **Single-step episodes** | Real underwriting involves multiple rounds of document review, clarification, negotiation |
| **No multi-agent interaction** | Real lending involves compliance officers, risk committees, applicants themselves |
| **Static tasks** | Only 5 fixed profiles — no randomization, procedural generation, or variability |
| **No state evolution** | The environment never "changes" — the profile is fixed, one decision, done |
| **No unit tests for grading edge cases** | You have test files but the grading logic could use more edge case coverage |
| **Hardcoded similarity maps** | The semantic similarity scores (0.99, 0.30, 0.01) are manually defined — not learned |

### What could be better

| Improvement | How |
|-------------|-----|
| **Randomized applicant generation** | Generate 100s of profiles procedurally instead of 5 fixed ones |
| **Multi-step episodes** | Agent asks for more documents, negotiates terms, then makes final decision |
| **Richer observation space** | Add employment history timeline, bank statements, transaction patterns |
| **Configurable difficulty** | Let the user tune how hard the grading is, or adjust weight parameters |
| **Metrics dashboard** | Track agent performance over time, not just single runs |

### What a Meta engineer might question

1. **"Why only 5 tasks?"** — More variety would better stress-test agent generalization.
2. **"Is the ground truth always correct?"** — The heuristic ground truth generator uses simple rules; real underwriting is more nuanced.
3. **"Why single-step?"** — The OpenEnv framework supports multi-step. Single-step limits what you can evaluate about agent reasoning.
4. **"How do you prevent overfitting?"** — With only 5 tasks, an agent could memorize all answers.
5. **"Where's the learning loop?"** — Your inference script runs the agent but never trains it.

---

## 9. 🎯 Round 2 Theme Analysis

### Theme 1 — Multi-Agent Interactions

**How your loan environment could use multiple agents:**

Imagine a real bank. A loan application doesn't go through one person — it goes through a *team*:

```
[Applicant Agent] ←→ [Loan Officer Agent] ←→ [Risk Committee Agent] ←→ [Compliance Agent]
```

- **Applicant Agent**: Tries to present the best case (might exaggerate income, argue for lower rates)
- **Loan Officer Agent**: Evaluates the application and writes a recommendation
- **Risk Committee Agent**: Reviews the officer's recommendation and may override it
- **Compliance Agent**: Checks if the decision follows regulations (e.g., fair lending laws)

**Real-world example**: In real banks, a loan over $500K must be approved by a committee, not a single officer. Each committee member may have different risk tolerance.

**What would change in your code:**
- `env.py` would need to accept multiple actions per step (one from each agent)
- A new `negotiation_round` system where agents can disagree and resolve
- New grading for *team coordination* — not just individual accuracy
- New observation that includes other agents' opinions

**Difficulty: Medium** — Your env structure is clean so adding agent roles is doable, but designing the interaction protocol requires careful thought.

---

### Theme 2 — Long-Horizon Planning

**How your environment could become multi-step:**

Instead of "here's a profile, give me a decision," imagine:

```
Step 1: Agent sees basic info (name, income, credit score)
   → Agent decides what to investigate next
Step 2: Agent requests employment verification
   → Environment reveals: "Self-employed, income varies ±30%"
Step 3: Agent requests debt details
   → Environment reveals: "Has 3 credit cards, 1 personal loan"
Step 4: Agent requests collateral appraisal
   → Environment reveals: "Property worth $200K"
Step 5: Agent makes final underwriting decision
```

**Real-world example**: A mortgage application takes 30-45 days. The officer doesn't see everything at once — they request documents one at a time, each revealing more information.

**What would change in your code:**
- `env.py` → `max_episode_steps` changes from 1 to 5-10
- New action type: "request information" alongside "make decision"
- Observation space expands progressively (new data unlocked each step)
- New reward component for *efficiency* — did the agent ask the right questions?
- `tasks.py` → Each task becomes a *sequence* of reveals, not one profile dump

**Difficulty: Medium** — Fits your existing architecture well. The `step()` function already returns `done` — you'd just set it to `False` for intermediate steps.

---

### Theme 3 — World Modeling

**How your agent could model the future:**

The agent could build a mental model of "what would happen if I approve this loan?"

```
Agent thinks:
  "If I approve Arjun at 14%+..."
     → P(default) = 78% (based on credit score 572 + 90% DTI)
     → Expected loss = $650K × 78% = $507K
     → Expected revenue = $650K × 14% × 10yr = $910K
     → But collection costs + risk = not worth it
  "If I reject Arjun..."
     → P(loss) = 0%
     → But we miss out on $910K potential revenue
  Decision: Reject (risk too high)
```

**Real-world example**: Insurance companies use "world models" to simulate 10,000 possible futures for each policyholder — what if they crash, get sick, live to 100? Banks do similar simulations.

**What would change in your code:**
- Add a `simulate_outcome()` function in the environment that predicts default probability
- Agent gets to "simulate" different decisions before committing
- New observation field: `simulated_outcomes` showing projected cash flows
- Reward could include a component for *quality of simulation* — did the agent predict correctly?

**Difficulty: Hard** — Requires building a financial simulation model. The environment itself needs to generate plausible futures.

---

### Theme 4 — Self-Improving Agent

**How your agent could learn from mistakes:**

After each loan decision, the agent gets feedback and adjusts its next decision:

```
Round 1: Agent evaluates Rajesh → Scores 0.76 (got risk right, decision wrong)
         Agent receives feedback: "You said Conditional Approve, should be Approve"
         Agent saves this to memory: "High credit + low DTI = Approve, not Conditional"

Round 2: Agent evaluates The Smiths (similar strong profile)
         Agent recalls: "Last time I was too cautious with a strong profile"
         Agent adjusts: → Approve (correct!)
         Score: 0.99 🎉
```

**Real-world example**: New loan officers make mistakes, get corrected by their manager, and gradually become more accurate. Your agent could do the same thing episode over episode.

**What would change in your code:**
- `inference.py` → Add a "memory" or "experience buffer" that stores past decisions + feedback
- The agent's prompt would include: "Here's what you got wrong last time"
- New reward component for *improvement rate* — is the agent getting better?
- Potentially: fine-tune the LLM on its own successful decisions

**Difficulty: Easy/Medium** — Your grading feedback is already detailed enough! You just need to feed it back to the agent across episodes. The `inference.py` loop already iterates over tasks in order.

---

## 10. ⭐ Best Theme Recommendation

### Quick Comparison

| Theme | Fit with your project | Ease of implementation | Judge impressiveness | Overall |
|-------|----------------------|----------------------|---------------------|---------|
| Multi-Agent | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🥈 |
| Long-Horizon | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🥇 |
| World Modeling | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 🥉 |
| Self-Improving | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 🥈 |

### 🏆 My Recommendation: **Long-Horizon Planning**

**Why this fits your project best:**

1. **Natural extension**: Loan underwriting in real life IS a multi-step process. Your current "single-step" env is the only thing that feels artificial.
2. **Leverages everything you built**: Your graders, reward function, models — all stay. You just add intermediate steps.
3. **Easy to explain**: "Instead of giving the agent everything at once, we reveal information gradually, and the agent decides what to ask for next."
4. **Demonstrates agent intelligence**: A multi-step agent that asks the right questions is more impressive than one that just reads a profile and answers.
5. **Differentiates from Round 1**: Judges will see clear growth — from "single snapshot decision" to "investigative process."

**Concrete implementation plan:**

```
Round 1 (what you have):
  reset() → Agent sees EVERYTHING → step() → done

Round 2 (proposed):
  reset() → Agent sees BASIC INFO only (name, income, credit score)
  step("request_employment_details") → Reveals job history
  step("request_debt_breakdown") → Reveals individual debts
  step("request_collateral_appraisal") → Reveals property value
  step("make_decision", {risk, decision, rate}) → Final decision → done

  New reward components:
    - Accuracy of final decision (what you already have)
    - Efficiency: Did the agent ask for ALL info, or skip what wasn't needed?
    - Ordering: Did the agent ask for the most important info first?
```

**Backup choice: Self-Improving Agent** — If Long-Horizon feels too much work, Self-Improving is the easiest to implement because your feedback system is already excellent. Just add a memory buffer to `inference.py`.

---

## 11. 📚 Key Concepts Explained (With Your Project Examples)

### 1. What is an OpenEnv environment?

**Simple**: It's a "test arena" for AI agents. Just like a gym has treadmills for humans, OpenEnv has challenges for AI.

**In your project**: Your `LoanUnderwritingEnv` is the arena. It serves up loan applications and grades the AI's decisions. It follows the standard `reset()` → `step()` → `state()` pattern so any compatible agent can plug in.

---

### 2. What is an AI agent?

**Simple**: An AI agent is a program that *observes* something, *thinks* about it, and *acts*. It's like a robot worker.

**In your project**: Your `inference.py` script IS the agent. It observes the applicant profile, sends it to an LLM (LLaMA 3.2), and acts by submitting a loan decision. The LLM is the "brain" of the agent.

---

### 3. What is a reward function?

**Simple**: It's the scoring system. Like getting gold stars in school. More stars = agent did better.

**In your project**: Your `compute_reward()` function in [rewards.py](file:///d:/Open_env_hackathon/loan-underwriting-openenv/environment/rewards.py) calculates a score from 0.01 to 0.99. If the agent gets the risk level, loan decision, and interest rate all correct AND they're consistent → high reward. If the agent gets everything wrong → low reward.

---

### 4. What is reinforcement learning?

**Simple**: It's learning by trial and error. Like a child touching a hot stove — they feel pain (negative reward), and learn not to do it again.

**In your project**: Your environment is DESIGNED for reinforcement learning. The agent makes a loan decision → gets a reward score → can use that feedback to make better decisions next time. Right now your agent doesn't "learn" between tasks (it's a one-shot LLM call), but the environment SUPPORTS an RL agent that would.

---

### 5. What is an observation space?

**Simple**: It's the "input" — everything the agent is allowed to see before making a decision. Like the information on a test paper.

**In your project**: Your observation space is defined in [models.py](file:///d:/Open_env_hackathon/loan-underwriting-openenv/environment/models.py#L97-L144) — it includes 12 fields about the applicant (name, age, income, credit score, etc.) plus the task description. The agent sees all of this in one big "packet."

---

### 6. What is an action space?

**Simple**: It's the set of possible "answers" the agent can give. Like multiple-choice options on a test.

**In your project**: Your action space is 3 choices:
- Risk Level: Low / Medium / High (3 options)
- Loan Decision: Approve / Conditional Approve / Reject (3 options)
- Interest Rate: 7-9% / 10-13% / 14%+ (3 options)

That's 3 × 3 × 3 = **27 possible combinations** the agent can submit.

---

### 7. What is a grader?

**Simple**: It's the automatic teacher that checks your answers and gives you a score.

**In your project**: Your [graders.py](file:///d:/Open_env_hackathon/loan-underwriting-openenv/environment/graders.py) has 4 grading functions:
- `grade_risk_level()` — checks the risk classification
- `grade_loan_decision()` — checks the approval decision
- `grade_interest_rate()` — checks the interest rate recommendation
- `grade_consistency()` — checks if all 3 make sense together

Each gives a score, and `grade_action()` combines them all.

---

### 8. What is semantic similarity?

**Simple**: It's measuring how "close" two answers are in *meaning*, not just whether they're the exact same word.

**In your project**: If the correct answer is "High" risk and the agent says "Medium" — that's *closer* (score: 0.30) than saying "Low" (score: 0.01). They're not the same word, but "Medium" is semantically closer to "High" than "Low" is. Your similarity maps capture this distance.

---

### 9. What is partial credit scoring?

**Simple**: Instead of "right or wrong" (100% or 0%), you can get 30% or 40% for being close. Like in school, showing your work gets you partial marks even if the final answer is wrong.

**In your project**: If the agent says "Conditional Approve" but the answer is "Approve" → it gets 0.40 (not zero). The agent was in the right ballpark — conditionally approving is much closer to approving than rejecting is.

---

### 10. What is multi-agent interaction?

**Simple**: Multiple AI agents working in the same environment, like players on a team (or opposing teams). They can cooperate, compete, or negotiate.

**In your project (future)**: Instead of one loan officer agent, you could have a loan officer + compliance checker + risk committee. Each agent sees different information and they negotiate toward a joint decision.

---

### 11. What is long-horizon planning?

**Simple**: Making a series of decisions over time, where each decision affects what happens next. Like playing chess — you plan 5 moves ahead, not just the next move.

**In your project (future)**: Instead of seeing the full profile and deciding immediately, the agent could investigate step-by-step — "first check credit score" → "then check employment" → "then check debt" → "finally decide." Each step has a cost (time), so the agent must plan which info to request.

---

### 12. What is world modeling?

**Simple**: Building a mental picture of how the world works, so you can predict what will happen before it does. Like a weather forecast — you model the atmosphere to predict rain.

**In your project (future)**: The agent could simulate: "If I approve this loan, what's the probability this person defaults in 2 years?" It builds an internal model of borrower behavior to predict outcomes.

---

### 13. What is self-improvement in AI?

**Simple**: The agent gets better at its job over time by remembering what worked and what didn't. Like studying for exams — you review your wrong answers to improve.

**In your project (future)**: After scoring 0.76 on Task 2, the agent remembers the feedback ("you were too cautious with medium risk applicants"). Next time it sees a similar profile, it adjusts its decision. This is the simplest thing to add — just feed the `grading_result.feedback` back into the LLM's prompt.

---

## 12. 📊 Visual ASCII Diagrams

### How Your Project Currently Works (Round 1)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    LOAN UNDERWRITING OPENENV                         │
│                        (Single-Step)                                 │
└──────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────┐
  │   Predefined Tasks   │
  │  ┌────────────────┐  │
  │  │ 1. Easy        │  │
  │  │ 2. Medium      │  │
  │  │ 3. Hard        │  │
  │  │ 4. Bankruptcy  │  │
  │  │ 5. Joint       │  │
  │  └────────────────┘  │
  └──────────┬──────────┘
             │ select task
             ▼
  ┌─────────────────────┐         ┌─────────────────────┐
  │   ENVIRONMENT        │         │   LLM AGENT          │
  │                      │         │   (inference.py)      │
  │  ┌────────────────┐  │ reset() │                      │
  │  │Applicant Profile├──┼────────►  Reads applicant     │
  │  │  • Name         │  │         │  profile data        │
  │  │  • Income       │  │         │                      │
  │  │  • Credit Score │  │         │  ┌────────────────┐  │
  │  │  • Debt         │  │         │  │  LLaMA 3.2 3B  │  │
  │  │  • Employment   │  │         │  │  (via HF API)  │  │
  │  │  • Loan Amount  │  │         │  └───────┬────────┘  │
  │  │  • Collateral   │  │         │          │           │
  │  │  • Defaults     │  │         │     Thinks...        │
  │  └────────────────┘  │         │          │           │
  │                      │  step() │          ▼           │
  │  ┌────────────────┐  │◄────────┤  Action:             │
  │  │   GRADER        │  │         │   Risk: Low          │
  │  │                 │  │         │   Decision: Approve   │
  │  │  Risk: 0.40×    │  │         │   Rate: 7-9%         │
  │  │  Decision: 0.35×│  │         │                      │
  │  │  Rate: 0.25×    │  │         └──────────────────────┘
  │  │  Consistency: ± │  │
  │  └───────┬────────┘  │
  │          │           │
  │          ▼           │
  │  ┌────────────────┐  │
  │  │  REWARD: 0.99   │  │
  │  │  Feedback: ✅✅✅ │  │
  │  │  Done: True     │  │
  │  └────────────────┘  │
  └──────────────────────┘
```

### How Round 2 Could Extend This (Long-Horizon)

```
┌──────────────────────────────────────────────────────────────────────┐
│               LOAN UNDERWRITING OPENENV v2                           │
│                    (Multi-Step Long-Horizon)                         │
└──────────────────────────────────────────────────────────────────────┘

  ┌────────────────┐
  │  Task Begins    │
  └───────┬────────┘
          │
          ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  STEP 1: Basic Profile Revealed                                │
  │  Agent sees: Name, Age, Income, Credit Score                   │
  │                                                                │
  │  Agent chooses: "REQUEST employment details"                   │
  │  Reward: 0 (no decision yet, just investigating)               │
  └───────┬────────────────────────────────────────────────────────┘
          │
          ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  STEP 2: Employment Details Revealed                           │
  │  Agent now sees: + Employment type, years, income stability    │
  │                                                                │
  │  Agent chooses: "REQUEST debt breakdown"                       │
  │  Reward: 0 (still investigating)                               │
  └───────┬────────────────────────────────────────────────────────┘
          │
          ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  STEP 3: Debt Details Revealed                                 │
  │  Agent now sees: + Existing debt, monthly expenses, defaults   │
  │                                                                │
  │  Agent chooses: "REQUEST collateral appraisal"                 │
  │  Reward: 0 (one more investigation)                            │
  └───────┬────────────────────────────────────────────────────────┘
          │
          ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  STEP 4: Collateral Info Revealed                              │
  │  Agent now sees: + Has collateral, property value              │
  │                                                                │
  │  Agent chooses: "MAKE DECISION"                                │
  │    → Risk: Medium / Decision: Conditional Approve / Rate: 10-13% │
  └───────┬────────────────────────────────────────────────────────┘
          │
          ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  FINAL GRADING                                                 │
  │                                                                │
  │  Decision Accuracy:  0.99 × 0.40 = 0.396                      │
  │  Decision Quality:   0.99 × 0.35 = 0.347                      │
  │  Rate Quality:       0.99 × 0.25 = 0.248                      │
  │  Consistency:        +0.10                                     │
  │  Efficiency Bonus:   +0.05 (asked only what was needed)        │
  │  ──────────────────────────────                                │
  │  TOTAL REWARD: 0.99                                            │
  │                                                                │
  │  Done: True ✅                                                  │
  └────────────────────────────────────────────────────────────────┘
```

### Comparison: Round 1 vs Round 2

```
ROUND 1 (Current)                 ROUND 2 (Proposed)
─────────────────                 ──────────────────

  ┌──────┐                          ┌──────┐
  │RESET │                          │RESET │
  └──┬───┘                          └──┬───┘
     │                                 │
     │ Full profile                    │ Basic info only
     │ revealed at once                │
     ▼                                 ▼
  ┌──────┐                          ┌──────┐
  │ STEP │ Make decision             │STEP 1│ Request info
  └──┬───┘                          └──┬───┘
     │                                 │
     │                                 ▼
     │                              ┌──────┐
     │                              │STEP 2│ Request more info
     │                              └──┬───┘
     │                                 │
     │                                 ▼
     │                              ┌──────┐
     │                              │STEP 3│ Request more info
     │                              └──┬───┘
     │                                 │
     │                                 ▼
     │                              ┌──────┐
     │                              │STEP 4│ Make decision
     │                              └──┬───┘
     │                                 │
     ▼                                 ▼
  ┌──────┐                          ┌──────┐
  │ DONE │                          │ DONE │
  │      │                          │      │
  │Score:│                          │Score: │
  │0.99  │                          │0.99 + │
  │      │                          │efficiency│
  └──────┘                          └──────┘

  1 step total                      4+ steps total
  Tests: knowledge                  Tests: knowledge +
                                    investigation strategy +
                                    efficiency
```

---

> [!TIP]
> **Quick Action Items for Round 2 Prep:**
> 1. Pick **Long-Horizon Planning** as your theme
> 2. Split your observation into "layers" that get revealed over multiple steps
> 3. Add a new action type: `"request_info"` alongside `"make_decision"`
> 4. Add an efficiency reward component (fewer steps to correct answer = higher bonus)
> 5. Keep all your existing grading and reward logic — it still works perfectly for the final step!

---

*Created for the Meta PyTorch OpenEnv Hackathon — Round 2 preparation*
