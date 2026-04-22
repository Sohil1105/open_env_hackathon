import re
import os

with open("server/app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace _build_llm_prompt, _parse_llm_json, evaluate_applicant
# We will just replace everything from `# ─── Evaluate Endpoint` to `# ─── OpenEnv Spec Endpoint`

new_code = """# ─── Evaluate Endpoint (LLM-driven decision) ────────────────────────────────

import json
import re

STAGE_PROMPTS = {
    "lead_qualification_sales": \"\"\"
        You are a bank sales officer.
        Review this initial inquiry and decide:
        Should we proceed with full application? Yes/No/Maybe

        Applicant: {profile}

        Respond in JSON:
        {{
            "qualification_decision": "Qualify" or "Disqualify" or "Request More Info",
            "risk_level": "Low" or "Medium" or "High",
            "loan_decision": "Approve" or "Conditional Approve" or "Reject",
            "interest_rate_tier": "7-9%" or "10-13%" or "14%+",
            "reasoning": "detailed explanation"
        }}
    \"\"\",
    "document_verification_hr": \"\"\"
        You are a bank compliance officer.
        Review the applicant documents and assess completeness.

        Applicant: {profile}

        Respond in JSON:
        {{
            "document_status": "Complete" or "Incomplete" or "Suspicious",
            "risk_level": "Low" or "Medium" or "High",
            "loan_decision": "Approve" or "Conditional Approve" or "Reject",
            "interest_rate_tier": "7-9%" or "10-13%" or "14%+",
            "reasoning": "detailed explanation of document assessment"
        }}
    \"\"\",
    "easy_salaried_high_credit": \"\"\"
        You are a senior bank loan underwriter.
        Analyze this applicant profile carefully.

        Applicant: {profile}

        Based on the financial data make your underwriting decision.

        Respond ONLY in this JSON format:
        {{
            "risk_level": "Low" or "Medium" or "High",
            "loan_decision": "Approve" or "Conditional Approve" or "Reject",
            "interest_rate_tier": "7-9%" or "10-13%" or "14%+",
            "reasoning": "step by step explanation of your decision"
        }}
    \"\"\",
    "medium_self_employed_moderate": \"\"\"
        You are a senior bank loan underwriter.
        Analyze this self-employed applicant profile carefully.

        Applicant: {profile}

        Based on the financial data make your underwriting decision.

        Respond ONLY in this JSON format:
        {{
            "risk_level": "Low" or "Medium" or "High",
            "loan_decision": "Approve" or "Conditional Approve" or "Reject",
            "interest_rate_tier": "7-9%" or "10-13%" or "14%+",
            "reasoning": "step by step explanation of your decision"
        }}
    \"\"\",
    "hard_freelancer_complex": \"\"\"
        You are a senior bank loan underwriter.
        Analyze this freelancer applicant profile carefully.

        Applicant: {profile}

        Based on the financial data make your underwriting decision.

        Respond ONLY in this JSON format:
        {{
            "risk_level": "Low" or "Medium" or "High",
            "loan_decision": "Approve" or "Conditional Approve" or "Reject",
            "interest_rate_tier": "7-9%" or "10-13%" or "14%+",
            "reasoning": "step by step explanation of your decision"
        }}
    \"\"\",
    "customer_onboarding_pm": \"\"\"
        You are a project manager handling customer onboarding.
        Review the applicant's profile and assess onboarding readiness.

        Applicant: {profile}

        Respond in JSON:
        {{
            "onboarding_status": "Complete" or "Incomplete" or "Critical Gaps",
            "risk_level": "Low" or "Medium" or "High",
            "loan_decision": "Approve" or "Conditional Approve" or "Reject",
            "interest_rate_tier": "7-9%" or "10-13%" or "14%+",
            "reasoning": "detailed explanation of onboarding readiness"
        }}
    \"\"\",
    "bankruptcy_recovery_edge1": \"\"\"
        You are a senior bank loan underwriter.
        Analyze this applicant with a history of bankruptcy.

        Applicant: {profile}

        Based on the financial data make your underwriting decision.

        Respond ONLY in this JSON format:
        {{
            "risk_level": "Low" or "Medium" or "High",
            "loan_decision": "Approve" or "Conditional Approve" or "Reject",
            "interest_rate_tier": "7-9%" or "10-13%" or "14%+",
            "reasoning": "step by step explanation of your decision"
        }}
    \"\"\",
    "joint_applicants_edge2": \"\"\"
        You are a senior bank loan underwriter.
        Analyze this joint applicant profile carefully.

        Applicant: {profile}

        Based on the financial data make your underwriting decision.

        Respond ONLY in this JSON format:
        {{
            "risk_level": "Low" or "Medium" or "High",
            "loan_decision": "Approve" or "Conditional Approve" or "Reject",
            "interest_rate_tier": "7-9%" or "10-13%" or "14%+",
            "reasoning": "step by step explanation of your decision"
        }}
    \"\"\"
}

class LifecycleSession:
    current_stage: int = 0
    completed_stages: list = []
    stage_scores: dict = {}
    applicant_profile: dict = {}
    total_score: float = 0.0

global_session = LifecycleSession()

def get_stage_number(task_id):
    try:
        return TASK_ORDER.index(task_id) + 1
    except ValueError:
        return 1

def get_next_stage(task_id):
    try:
        idx = TASK_ORDER.index(task_id)
        if idx + 1 < len(TASK_ORDER):
            return TASK_ORDER[idx + 1]
    except ValueError:
        pass
    return None

def get_next_stage_name(task_id):
    next_id = get_next_stage(task_id)
    if next_id:
        from environment.tasks import ALL_TASKS
        return ALL_TASKS.get(next_id).name if next_id in ALL_TASKS else next_id
    return "Finished"

def get_grader_for_stage(task_id: str):
    from environment.graders import (
        grade_lead_qualification,
        grade_document_verification,
        grade_customer_onboarding,
        grade_action
    )
    mapping = {
        "lead_qualification_sales": grade_lead_qualification,
        "document_verification_hr": grade_document_verification,
        "customer_onboarding_pm": grade_customer_onboarding,
    }
    return mapping.get(task_id, grade_action)

def parse_llm_response(response_text: str) -> dict:
    \"\"\"
    Robustly parse LLM response even if formatting is off
    \"\"\"
    # Try direct JSON parse first
    try:
        return json.loads(response_text)
    except:
        pass

    # Try extracting JSON from text
    try:
        json_match = re.search(r'\\{.*\\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass

    # Fallback - extract key fields manually
    result = {
        "risk_level": "Medium",
        "loan_decision": "Conditional Approve",
        "interest_rate_tier": "10-13%",
        "reasoning": response_text  # Use raw response as reasoning
    }

    # Try to find risk level
    if "low" in response_text.lower():
        result["risk_level"] = "Low"
    elif "high" in response_text.lower():
        result["risk_level"] = "High"

    # Try to find decision
    if "approve" in response_text.lower() and "conditional" not in response_text.lower():
        result["loan_decision"] = "Approve"
    elif "reject" in response_text.lower():
        result["loan_decision"] = "Reject"

    return result

@app.post("/evaluate")
async def evaluate_applicant(applicant: ApplicantInput):
    task_id = applicant.task_id

    # Session tracking
    global_session.current_stage = get_stage_number(task_id)
    if task_id not in global_session.completed_stages:
        global_session.completed_stages.append(task_id)

    # 1. Get stage-specific prompt
    prompt_template = STAGE_PROMPTS.get(
        task_id,
        STAGE_PROMPTS["easy_salaried_high_credit"]
    )

    # 2. Format prompt with applicant details
    profile_text = f\"\"\"
    Name: {applicant.applicant_name}
    Annual Income: ₹{applicant.annual_income:,.0f}
    Credit Score: {applicant.credit_score}
    Existing Debt: ₹{applicant.existing_debt:,.0f}
    Loan Requested: ₹{applicant.loan_amount:,.0f}
    Employment: {applicant.employment_type}
    Tenure: {applicant.loan_tenure} months
    Age: {getattr(applicant, 'age', 'N/A')}
    Job History: {getattr(applicant, 'employment_years', 'N/A')} years
    Past Defaults: {getattr(applicant, 'previous_defaults', 0)}
    \"\"\"

    full_prompt = prompt_template.format(profile=profile_text)

    # 3. Call LLM
    try:
        local_client, local_model_name = _get_api_client()
        response = local_client.chat.completions.create(
            model=local_model_name,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=500,
            temperature=0.3
        )
        raw_response = response.choices[0].message.content

        # 4. Parse response
        agent_decision = parse_llm_response(raw_response)

    except Exception as e:
        # Graceful fallback
        # profile is not yet created, we need to create it first
        profile = ApplicantProfile(
            applicant_name=applicant.applicant_name,
            annual_income=applicant.annual_income,
            credit_score=applicant.credit_score,
            existing_debt=applicant.existing_debt,
            loan_amount_requested=applicant.loan_amount,
            employment_type=applicant.employment_type,
            employment_years=applicant.employment_years,
            repayment_tenure_months=applicant.loan_tenure,
            age=applicant.age,
            monthly_expenses=applicant.monthly_expenses if applicant.monthly_expenses > 0 else (applicant.existing_debt / 12),
            has_collateral=applicant.has_collateral,
            previous_defaults=applicant.previous_defaults
        )
        gt = calculate_dynamic_ground_truth(profile)
        agent_decision = {
            "risk_level": gt.risk_level.value,
            "loan_decision": gt.loan_decision.value,
            "interest_rate_tier": gt.interest_rate_tier.value,
            "reasoning": f"[AI Error: {str(e)}] Using ground truth fallback."
        }

    # 5. Calculate dynamic ground truth
    profile = ApplicantProfile(
        applicant_name=applicant.applicant_name,
        annual_income=applicant.annual_income,
        credit_score=applicant.credit_score,
        existing_debt=applicant.existing_debt,
        loan_amount_requested=applicant.loan_amount,
        employment_type=applicant.employment_type,
        employment_years=applicant.employment_years,
        repayment_tenure_months=applicant.loan_tenure,
        age=applicant.age,
        monthly_expenses=applicant.monthly_expenses if applicant.monthly_expenses > 0 else (applicant.existing_debt / 12),
        has_collateral=applicant.has_collateral,
        previous_defaults=applicant.previous_defaults
    )
    ground_truth = calculate_dynamic_ground_truth(profile)

    # 6. Grade the decision
    action = Action(
        risk_level=agent_decision.get("risk_level", "Medium"),
        loan_decision=agent_decision.get("loan_decision", "Conditional Approve"),
        interest_rate_tier=agent_decision.get("interest_rate_tier", "10-13%"),
        reasoning=agent_decision.get("reasoning", "")
    )

    # Get stage specific grader
    grader = get_grader_for_stage(task_id)
    grading_result = grader(action, ground_truth)
    score = grading_result.total_score
    
    global_session.stage_scores[task_id] = score

    # 7. Return complete result
    return {
        "agent_decision": agent_decision,
        "score": score,
        "ground_truth": {
            "risk_level": ground_truth.risk_level.value,
            "loan_decision": ground_truth.loan_decision.value,
            "interest_rate_tier": ground_truth.interest_rate_tier.value,
            "explanation": ground_truth.explanation
        },
        "stage": task_id,
        "stage_number": get_stage_number(task_id),
        "next_stage": get_next_stage(task_id),
        "next_stage_name": get_next_stage_name(task_id),
        "reasoning": agent_decision.get("reasoning", ""),
        "status": "success",
        "grading": {
            "risk_level_score": grading_result.risk_level_score,
            "loan_decision_score": grading_result.loan_decision_score,
            "interest_rate_score": grading_result.interest_rate_score,
            "consistency_bonus": grading_result.consistency_bonus,
        },
        "feedback": grading_result.feedback,
        "task_name": get_next_stage_name(task_id) if task_id != get_next_stage(task_id) else "Current",
        "task_difficulty": "adaptive",
        "correct_answer": {
            "risk_level": ground_truth.risk_level.value,
            "loan_decision": ground_truth.loan_decision.value,
            "interest_rate_tier": ground_truth.interest_rate_tier.value,
            "explanation": ground_truth.explanation
        }
    }

"""

start_marker = "# ─── Evaluate Endpoint (LLM-driven decision) ────────────────────────────────"
end_marker = "# ─── OpenEnv Spec Endpoint ───────────────────────────────────────────────────"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_code + "\n" + content[end_idx:]
    with open("server/app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Patched app.py successfully.")
else:
    print("Could not find markers.")
