"""
Baseline LLM Agent for the Loan Underwriting OpenEnv Environment.

This script:
1. Instantiates the LoanUnderwritingEnv environment
2. Iterates through all 3 tasks (easy → medium → hard)
3. For each task, sends the applicant profile to an LLM via OpenAI-compatible API
4. Parses the LLM's response into a structured Action
5. Steps the environment and collects scores
6. Prints a detailed summary of scores per task

Environment Variables Required:
- API_BASE_URL: Base URL for the OpenAI-compatible API endpoint
- MODEL_NAME: Name of the model to use for inference
- HF_TOKEN: Hugging Face token used as the API key

Designed to complete in under 20 minutes on 2 vCPU / 8 GB RAM.
"""

import os
import json
import re
import time
import traceback

from openai import OpenAI

from environment import (
    LoanUnderwritingEnv,
    Action,
    RiskLevel,
    LoanDecision,
    InterestRateTier,
    TASK_ORDER,
)
from environment.rewards import format_reward_breakdown

# ─── Client Setup (exact spec from requirements) ─────────────────────────────

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("HF_TOKEN", "dummy_token"),
)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")


# ─── Prompt Construction ─────────────────────────────────────────────────────

def build_system_prompt() -> str:
    """Build the system prompt that instructs the LLM on its role."""
    return """You are an expert bank loan underwriter with 20 years of experience in risk assessment.
You evaluate loan applications by analyzing applicant profiles and making three key decisions:

1. **Risk Level Classification**: Low, Medium, or High
2. **Loan Decision**: Approve, Conditional Approve, or Reject
3. **Interest Rate Tier**: 7-9%, 10-13%, or 14%+

Your decisions must be logically consistent:
- Low risk → typically Approve at 7-9%
- Medium risk → typically Conditional Approve at 10-13%
- High risk → typically Reject (or Conditional Approve at 14%+ in rare cases)

You MUST respond with a valid JSON object containing exactly these fields:
{
    "risk_level": "Low" | "Medium" | "High",
    "loan_decision": "Approve" | "Conditional Approve" | "Reject",
    "interest_rate_tier": "7-9%" | "10-13%" | "14%+",
    "reasoning": "Brief explanation of your decision"
}

Respond ONLY with the JSON object. No other text."""


def build_user_prompt(observation: dict) -> str:
    """Build the user prompt containing the applicant profile."""
    # Calculate key ratios for the agent
    annual_income = observation["annual_income"]
    existing_debt = observation["existing_debt"]
    loan_amount = observation["loan_amount_requested"]
    monthly_income = annual_income / 12
    monthly_expenses = observation["monthly_expenses"]

    dti_ratio = (existing_debt / annual_income * 100) if annual_income > 0 else float("inf")
    lti_ratio = (loan_amount / annual_income * 100) if annual_income > 0 else float("inf")
    monthly_disposable = monthly_income - monthly_expenses

    prompt = f"""## Loan Application for Review

**Task:** {observation['task_description']}

### Applicant Profile

| Field | Value |
|-------|-------|
| Name | {observation['applicant_name']} |
| Age | {observation['age']} years |
| Annual Income | ${annual_income:,.2f} |
| Monthly Income | ${monthly_income:,.2f} |
| Credit Score | {observation['credit_score']} |
| Employment Type | {observation['employment_type']} |
| Employment Tenure | {observation['employment_years']} years |
| Monthly Expenses | ${monthly_expenses:,.2f} |
| Monthly Disposable Income | ${monthly_disposable:,.2f} |
| Existing Debt | ${existing_debt:,.2f} |
| Debt-to-Income Ratio | {dti_ratio:.1f}% |
| Loan Amount Requested | ${loan_amount:,.2f} |
| Loan-to-Income Ratio | {lti_ratio:.1f}% |
| Repayment Tenure | {observation['repayment_tenure_months']} months |
| Has Collateral | {'Yes' if observation['has_collateral'] else 'No'} |
| Previous Defaults | {observation['previous_defaults']} |

### Key Risk Indicators
- Debt-to-Income Ratio: {dti_ratio:.1f}% {'(LOW - Good)' if dti_ratio < 30 else '(MODERATE)' if dti_ratio < 50 else '(HIGH - Risky)'}
- Credit Score: {observation['credit_score']} {'(EXCELLENT 750+)' if observation['credit_score'] >= 750 else '(GOOD 700-749)' if observation['credit_score'] >= 700 else '(FAIR 620-699)' if observation['credit_score'] >= 620 else '(POOR <620)'}
- Loan-to-Income Ratio: {lti_ratio:.1f}% {'(Manageable)' if lti_ratio < 80 else '(Stretched)' if lti_ratio < 150 else '(Overextended)'}
- Previous Defaults: {observation['previous_defaults']} {'(Clean record)' if observation['previous_defaults'] == 0 else '(Concerning)'}

Please analyze this application and provide your underwriting decision as a JSON object."""

    return prompt


# ─── Response Parsing ─────────────────────────────────────────────────────────

def parse_llm_response(response_text: str) -> Action:
    """
    Parse the LLM's response text into a structured Action.

    Handles various response formats:
    - Clean JSON
    - JSON wrapped in markdown code blocks
    - JSON with extra text before/after

    Falls back to regex extraction if JSON parsing fails.
    """
    # Try to extract JSON from the response
    text = response_text.strip()

    # Remove markdown code block wrappers if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Try direct JSON parsing
    try:
        data = json.loads(text)
        return Action(
            risk_level=data.get("risk_level", "Medium"),
            loan_decision=data.get("loan_decision", "Conditional Approve"),
            interest_rate_tier=data.get("interest_rate_tier", "10-13%"),
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError):
        pass

    # Try to find a JSON object within the text using regex
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return Action(
                risk_level=data.get("risk_level", "Medium"),
                loan_decision=data.get("loan_decision", "Conditional Approve"),
                interest_rate_tier=data.get("interest_rate_tier", "10-13%"),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError):
            pass

    # Last resort: try to extract individual fields via regex
    risk_match = re.search(r'"risk_level"\s*:\s*"(Low|Medium|High)"', text, re.IGNORECASE)
    decision_match = re.search(
        r'"loan_decision"\s*:\s*"(Approve|Conditional Approve|Reject)"', text, re.IGNORECASE
    )
    rate_match = re.search(
        r'"interest_rate_tier"\s*:\s*"(7-9%|10-13%|14%\+?)"', text, re.IGNORECASE
    )

    return Action(
        risk_level=risk_match.group(1) if risk_match else "Medium",
        loan_decision=decision_match.group(1) if decision_match else "Conditional Approve",
        interest_rate_tier=rate_match.group(1) if rate_match else "10-13%",
        reasoning=f"Parsed from unstructured response: {text[:200]}",
    )


# ─── LLM Agent ───────────────────────────────────────────────────────────────

def run_agent(observation: dict) -> Action:
    """
    Send the observation to the LLM and get an underwriting decision.

    Args:
        observation: Dictionary representation of the Observation

    Returns:
        Parsed Action from the LLM's response
    """
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(observation)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent, analytical responses
            max_tokens=512,
        )

        response_text = response.choices[0].message.content
        print(f"  LLM Response: {response_text[:300]}...")
        return parse_llm_response(response_text)

    except Exception as e:
        print(f"  ⚠️ LLM API error: {e}")
        traceback.print_exc()
        # Return a safe default action on API failure
        return Action(
            risk_level=RiskLevel.MEDIUM,
            loan_decision=LoanDecision.CONDITIONAL_APPROVE,
            interest_rate_tier=InterestRateTier.MEDIUM,
            reasoning=f"Default action due to API error: {str(e)}",
        )


# ─── Main Execution ──────────────────────────────────────────────────────────

def main():
    """
    Main entry point: runs the agent through all 3 tasks and prints scores.
    """
    print("=" * 70)
    print("🏦 LOAN UNDERWRITING OPENENV — BASELINE LLM AGENT")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {os.environ.get('API_BASE_URL', 'NOT SET')}")
    print()

    # Initialize the environment
    env = LoanUnderwritingEnv()
    total_start_time = time.time()

    # Track scores across all tasks
    task_scores = {}
    total_score = 0.0

    # Iterate through all tasks in order
    for i, task_id in enumerate(TASK_ORDER, 1):
        print(f"{'─' * 70}")
        print(f"📋 TASK {i}/3: {task_id}")
        print(f"{'─' * 70}")

        task_start_time = time.time()

        # Reset environment for this task
        state = env.reset(task_id)
        observation = state.observation

        print(f"  Difficulty: {observation.task_difficulty.upper()}")
        print(f"  Applicant: {observation.applicant_name}")
        print(f"  Credit Score: {observation.credit_score}")
        print(f"  Income: ${observation.annual_income:,.2f}")
        print(f"  Loan Request: ${observation.loan_amount_requested:,.2f}")
        print()

        # Get the agent's decision from the LLM
        print("  🤖 Querying LLM for underwriting decision...")
        observation_dict = observation.model_dump()
        action = run_agent(observation_dict)

        print(f"\n  Agent Decision:")
        print(f"    Risk Level: {action.risk_level.value}")
        print(f"    Loan Decision: {action.loan_decision.value}")
        print(f"    Interest Rate: {action.interest_rate_tier.value}")
        if action.reasoning:
            reasoning_preview = action.reasoning[:150]
            print(f"    Reasoning: {reasoning_preview}...")
        print()

        # Step the environment
        state, reward, done, info = env.step(action)

        # Display results
        print(f"  📊 GRADING RESULTS:")
        print(f"  {info['reward_breakdown']}")
        print(f"\n  {info['feedback']}")

        task_duration = time.time() - task_start_time
        print(f"\n  ⏱️ Task completed in {task_duration:.1f}s")

        # Store scores
        task_scores[task_id] = {
            "score": reward,
            "grading": info["grading"],
            "duration": task_duration,
        }
        total_score += reward
        print()

    # ─── Final Summary ────────────────────────────────────────────────────

    total_duration = time.time() - total_start_time

    print("============================================")
    print(" LOAN UNDERWRITING OPENENV - INFERENCE RUN")
    print("============================================")

    display_names = {
        "easy_salaried_high_credit": "Easy",
        "medium_self_employed_moderate": "Medium",
        "hard_freelancer_complex": "Hard",
        "bankruptcy_recovery_edge1": "Edge 1",
        "joint_applicants_edge2": "Edge 2",
    }

    for i, (task_id, result) in enumerate(task_scores.items(), 1):
        score = result["score"]
        name = display_names.get(task_id, "Unknown")
        task_str = f"Task {i}/{len(TASK_ORDER)} ({name})"
        print(f"{task_str:<18}| Score: {score:.2f} ✅")

    average_score = total_score / len(TASK_ORDER)

    print("─────────────────────────────────────────")
    print(f"Average Score: {average_score:.2f}/1.0")
    print(f"Total Time:    {total_duration:.1f} seconds")
    print("============================================")

    # Validate all scores are in [0.0, 1.0]
    all_valid = all(
        0.0 <= result["score"] <= 1.0 for result in task_scores.values()
    )
    if not all_valid:
        print("  ❌ WARNING: Some scores are outside valid range!")

    return task_scores


if __name__ == "__main__":
    main()
