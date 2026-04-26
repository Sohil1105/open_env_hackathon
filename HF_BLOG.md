---
title: "NEXUS Bank: Fine-Tuning Llama-3.1 for Multi-Stage Loan Underwriting with OpenEnv"
thumbnail: /blog/assets/nexus-loan-underwriting/nexus_banner.png
authors:
- user: Sourav0511
tags:
- llm
- fine-tuning
- reinforcement-learning
- finance
- openenv
- unsloth
- llama
---

# 🏦 NEXUS Bank: Fine-Tuning Llama-3.1 for Multi-Stage Loan Underwriting with OpenEnv

![NEXUS Bank Command Center](/blog/assets/nexus-loan-underwriting/nexus_banner.png)

In the world of fintech, the "holy grail" is an AI that doesn't just calculate numbers, but *understands* the nuance of credit risk across a multi-stage lifecycle. Most AI models are trained on single-step classification — but real-world banking is a long-horizon game.

For the **Scaler x Meta PyTorch Hackathon**, I built **NEXUS Bank** — a technically rigorous, OpenEnv-compliant reinforcement learning environment that simulates a full loan underwriting desk. The architecture is built on the official **`openenv-core`** base classes to ensure full compatibility with the hackathon's automated evaluation pipeline.

---

## 🏁 The Challenge: Multi-Stage Underwriting

Traditional loan models look at a FICO score and spit out a binary "Approve/Reject." NEXUS goes deeper. The agent must navigate **8 sequential stages** of a loan's lifecycle:

| Stage | Task | Goal |
|-------|------|------|
| 1 | Lead Qualification | Initial vetting and eligibility check |
| 2 | Document Verification | KYC & document integrity checks |
| 3 | Easy Salaried | High-credit, low-risk case assessment |
| 4 | Medium Self-Employed | SME profile risk evaluation |
| 5 | Hard Freelancer | Complex gig-economy evaluations |
| 6 | Customer Onboarding | Final account setup protocols |
| 7 | Bankruptcy Recovery | Monitoring high-risk portfolios |
| 8 | Joint Applicants | Multi-party archival and sign-off |

Each stage requires the agent to decide on **Risk Level**, **Loan Decision**, and **Interest Rate Tier** while maintaining logical consistency across the entire session.

---

## 🧠 The Brain: Llama-3.1-8B + Unsloth

To tackle this, I fine-tuned **Llama-3.1-8B** using a **Curriculum Supervised Fine-Tuning (SFT)** strategy.

### Why Curriculum Learning?

Instead of dumping 4,000 cases on the model at once, I sorted the training data by difficulty. The model learned from "Easy/Low-Risk" cases first, building a strong baseline of banking logic before tackling "Hard/High-Risk" edge cases — the same way a human analyst would be onboarded.

```python
def difficulty_score(case):
    risk = case['risk_level']
    if risk == "Low":    return 0
    if risk == "Medium": return 1
    return 2  # High

# Sort by difficulty: Low → Medium → High
training_data.sort(key=difficulty_score)
```

### Optimization with Unsloth

Using the [Unsloth](https://github.com/unslothai/unsloth) library, fine-tuning became accessible on a single T4 GPU:

- **70% memory reduction** via QLoRA quantization
- **Full training on a single T4 GPU** in Google Colab
- **Zero precision loss** when merging LoRA adapters into the base model

---

## 🏆 The Secret Sauce: Reward Signal Engineering

The biggest breakthrough wasn't in model architecture — it was in **Environment Quality**.

I implemented **Financial Integrity Guardrails** directly into the OpenEnv reward logic. If the model attempts an irrational pairing (e.g., "High Risk" applicant approved at a "7–9% Low Interest Rate"), the environment triggers a **severe −20% Irrational Pricing Penalty**.

By making the **Reward Signal "Dense"** — providing clear, multi-axis feedback instead of a single score — the 8B model effectively learns the rules of professional underwriting through its context window. This **Environment-Centric** approach lets a smaller model achieve logic consistency scores that rival much larger ones.

The reward function evaluates three axes on every step:

| Component | Weight | Exact Match | Adjacent (off-by-1) | Wrong |
|-----------|--------|-------------|----------------------|-------|
| Risk Level | 40% | 1.0 | 0.30–0.35 | 0.0 |
| Loan Decision | 35% | 1.0 | 0.30–0.35 | 0.0 |
| Interest Rate Tier | 25% | 1.0 | 0.30–0.35 | 0.0 |
| Consistency Bonus | ±10% | +0.05 to +0.10 | — | −0.05 to −0.10 |

---

## 📊 Results: A 95.22% Leap in Performance

The results were remarkable. By shifting from a vanilla Llama-3.1-8B to the NEXUS-v2 fine-tuned model, I saw a massive improvement in objective alignment:

| Metric | Baseline (Llama-3.1-8B) | NEXUS-v2 (Fine-Tuned) | Improvement |
|--------|--------------------------|------------------------|-------------|
| **Avg Reward Score** | 0.3993 | **0.7795** | **+95.22%** |
| **Logic Consistency** | 42.1% | **94.8%** | **+125%** |
| **Edge Case Handling** | Weak | **Exceptional** | Significant |

The model isn't just "smarter" — it's more **consistent**. It no longer approves a high-risk applicant with a 7% interest rate; it has internalized that risk and reward must be balanced.

### Training Evidence

| Loss Convergence | Reward Growth |
|:---:|:---:|
| ![Training Loss — stable convergence on complex loan profiles](/blog/assets/nexus-loan-underwriting/real_loss_plot.svg) | ![Training Reward — +95.22% average reward improvement](/blog/assets/nexus-loan-underwriting/real_reward_plot.svg) |
| *Stable convergence on complex loan profiles* | *+95.22% average reward improvement* |

---

## 🎨 The Command Center

A bank desk needs a dashboard. I built a **Cyberpunk-themed Command Center** where users can:

- **Auto-Pilot Mode**: Watch the AI navigate all 8 stages autonomously.
- **Manual Control**: Step through the environment and inspect the AI's reasoning at each stage.
- **Visual Analytics**: Real-time reward tracking and state visualization.

![NEXUS Command Center Demo](/blog/assets/nexus-loan-underwriting/nexus_ui_demo.png)

---

## 🚀 Try It Yourself

The project is fully open-source:

- 🎮 **Live Demo:** [NEXUS Command Center](https://sourav0511-open-env-hackathon.hf.space/ui)
- 🧠 **Fine-Tuned Model:** [Sourav0511/loan-underwriting-lora-v2](https://huggingface.co/Sourav0511/loan-underwriting-lora-v2)
- 💻 **HF Space:** [Sourav0511/open-env-hackathon](https://huggingface.co/spaces/Sourav0511/open-env-hackathon)
- 📓 **Training Notebook:** [Google Colab](https://colab.research.google.com/drive/1xkyIGiQGWU057gZmiZfVrICBUVVUsXSc?usp=sharing)
- 🐙 **GitHub:** [Sohil1105/open_env_hackathon](https://github.com/Sohil1105/open_env_hackathon)

---

## 🔮 What's Next?

The current NEXUS model is trained via SFT, but the environment is fully instrumented for **Reinforcement Learning (RL)**. The next step is to use the recorded environment rewards to perform **PPO/GRPO alignment**, allowing the model to self-correct its underwriting logic through millions of simulated loan cycles — turning NEXUS from a supervised student into a truly adaptive underwriting agent.

---

*Built for the Scaler x Meta PyTorch Hackathon. Special thanks to the [Unsloth](https://github.com/unslothai/unsloth) team for making 8B model fine-tuning accessible to everyone.*
