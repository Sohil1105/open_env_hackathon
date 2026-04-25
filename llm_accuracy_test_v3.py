"""
LLM Accuracy Test v3 — lora-v2 model on localhost:8000
25 cases: 15 blind (Option A) + 10 edge (Option B)
Saves results to llm_test_results_v3.json
"""

import sys
import os
import json
import time
import requests
from dataclasses import dataclass
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from environment.graders import calculate_dynamic_ground_truth
from environment.models import ApplicantProfile, EmploymentType

BASE_URL = "http://localhost:8000"

# ─── Option A: 15 Blind Test Cases ───────────────────────────────────────────

OPTION_A_CASES = [
    {"id": 1,  "fico": 780, "dti": 10, "defaults": 0, "income": 120000, "purpose": "home improvement",  "exp_risk": "Low",         "exp_decision": "Approve"},
    {"id": 2,  "fico": 720, "dti": 19, "defaults": 0, "income": 95000,  "purpose": "debt consolidation","exp_risk": "Low",         "exp_decision": "Approve"},
    {"id": 3,  "fico": 695, "dti": 25, "defaults": 0, "income": 75000,  "purpose": "medical",           "exp_risk": "Medium",      "exp_decision": "Conditional Approve"},
    {"id": 4,  "fico": 665, "dti": 32, "defaults": 0, "income": 60000,  "purpose": "education",         "exp_risk": "Medium",      "exp_decision": "Conditional Approve"},
    {"id": 5,  "fico": 640, "dti": 40, "defaults": 1, "income": 48000,  "purpose": "small business",    "exp_risk": "Medium-High", "exp_decision": "Conditional Approve"},
    {"id": 6,  "fico": 610, "dti": 50, "defaults": 0, "income": 55000,  "purpose": "car",               "exp_risk": "High",        "exp_decision": "Reject"},
    {"id": 7,  "fico": 580, "dti": 60, "defaults": 2, "income": 35000,  "purpose": "vacation",          "exp_risk": "High",        "exp_decision": "Reject"},
    {"id": 8,  "fico": 750, "dti": 45, "defaults": 0, "income": 110000, "purpose": "home improvement",  "exp_risk": "Medium",      "exp_decision": "Conditional Approve"},
    {"id": 9,  "fico": 800, "dti": 8,  "defaults": 0, "income": 200000, "purpose": "business expansion","exp_risk": "Low",         "exp_decision": "Approve"},
    {"id": 10, "fico": 630, "dti": 15, "defaults": 0, "income": 42000,  "purpose": "debt consolidation","exp_risk": "Medium",      "exp_decision": "Conditional Approve"},
    {"id": 11, "fico": 720, "dti": 35, "defaults": 1, "income": 80000,  "purpose": "medical",           "exp_risk": "Medium",      "exp_decision": "Conditional Approve"},
    {"id": 12, "fico": 550, "dti": 70, "defaults": 3, "income": 28000,  "purpose": "personal",          "exp_risk": "High",        "exp_decision": "Reject"},
    {"id": 13, "fico": 740, "dti": 22, "defaults": 0, "income": 90000,  "purpose": "home improvement",  "exp_risk": "Low",         "exp_decision": "Approve"},
    {"id": 14, "fico": 670, "dti": 28, "defaults": 0, "income": 65000,  "purpose": "education",         "exp_risk": "Medium",      "exp_decision": "Conditional Approve"},
    {"id": 15, "fico": 600, "dti": 55, "defaults": 2, "income": 32000,  "purpose": "small business",    "exp_risk": "High",        "exp_decision": "Reject"},
]

# ─── Option B: 10 Edge Cases ─────────────────────────────────────────────────

OPTION_B_CASES = [
    {"id": "E1",  "fico": 719, "dti": 19, "defaults": 0, "income": 80000,  "note": "1 below Low threshold"},
    {"id": "E2",  "fico": 720, "dti": 20, "defaults": 0, "income": 80000,  "note": "Exact boundary"},
    {"id": "E3",  "fico": 660, "dti": 35, "defaults": 0, "income": 70000,  "note": "Both at boundary"},
    {"id": "E4",  "fico": 750, "dti": 48, "defaults": 0, "income": 100000, "note": "High DTI, great FICO"},
    {"id": "E5",  "fico": 620, "dti": 19, "defaults": 0, "income": 60000,  "note": "Low FICO, great DTI"},
    {"id": "E6",  "fico": 680, "dti": 28, "defaults": 1, "income": 75000,  "note": "1 default, mid profile"},
    {"id": "E7",  "fico": 800, "dti": 5,  "defaults": 0, "income": 200000, "note": "Perfect profile"},
    {"id": "E8",  "fico": 500, "dti": 80, "defaults": 5, "income": 25000,  "note": "Worst possible"},
    {"id": "E9",  "fico": 700, "dti": 33, "defaults": 0, "income": 65000,  "note": "Borderline medium"},
    {"id": "E10", "fico": 645, "dti": 42, "defaults": 1, "income": 55000,  "note": "Triple borderline"},
]


def build_payload(fico, dti_pct, defaults, income, purpose="general", case_id=None):
    existing_debt = (dti_pct / 100.0) * income
    loan_amount = income * 0.5
    name = f"Test Case {case_id}" if case_id else "Test Applicant"
    return {
        "applicant_name": name,
        "annual_income": float(income),
        "credit_score": int(fico),
        "existing_debt": float(existing_debt),
        "loan_amount": float(loan_amount),
        "employment_type": "salaried",
        "employment_years": 5.0,
        "job_history_years": 5.0,
        "loan_tenure": 60,
        "task_id": "easy_salaried_high_credit",
        "age": 35,
        "monthly_expenses": float(income * 0.2 / 12),
        "has_collateral": False,
        "previous_defaults": int(defaults),
        "loan_purpose": purpose,
        "public_records": 0,
        "credit_inquiries_6mo": 0,
        "documents_submitted": ["pay_stub", "bank_statement", "id_proof"],
    }


def call_evaluate(payload, retries=2):
    for attempt in range(retries + 1):
        try:
            resp = requests.post(f"{BASE_URL}/evaluate", json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            if attempt < retries:
                print(f"    [timeout, retrying {attempt+1}/{retries}]")
                time.sleep(3)
            else:
                return {"error": "timeout", "agent_decision": {}}
        except Exception as e:
            return {"error": str(e), "agent_decision": {}}


def normalize_risk(val: str) -> str:
    v = val.strip().lower()
    if v in ("low",):
        return "Low"
    if v in ("medium", "medium-high", "moderate"):
        return "Medium"
    if v in ("high",):
        return "High"
    return val.title()


def normalize_decision(val: str) -> str:
    v = val.strip().lower()
    if "conditional" in v:
        return "Conditional Approve"
    if "approve" in v:
        return "Approve"
    if "reject" in v or "deny" in v:
        return "Reject"
    return val.title()


def score_case(llm_risk, llm_decision, exp_risk, exp_decision):
    exp_risk_norm = "Medium" if exp_risk in ("Medium-High",) else exp_risk
    risk_match = normalize_risk(llm_risk) == exp_risk_norm
    dec_match = normalize_decision(llm_decision) == exp_decision
    if risk_match and dec_match:
        return "CORRECT", 1.0
    elif risk_match or dec_match:
        return "PARTIAL", 0.5
    else:
        return "INCORRECT", 0.0


def build_profile_for_grader(fico, dti_pct, defaults, income, case_id="test"):
    existing_debt = (dti_pct / 100.0) * income
    loan_amount = income * 0.5
    return ApplicantProfile(
        applicant_name=f"Edge Case {case_id}",
        age=35,
        annual_income=float(income),
        credit_score=int(fico),
        existing_debt=float(existing_debt),
        employment_type=EmploymentType.SALARIED,
        employment_years=5.0,
        loan_amount_requested=float(loan_amount),
        repayment_tenure_months=60,
        monthly_expenses=float(income * 0.2 / 12),
        has_collateral=False,
        previous_defaults=int(defaults),
        documents_submitted=["pay_stub", "bank_statement", "id_proof"],
    )


def run_option_a():
    print("\n" + "=" * 70)
    print("  OPTION A — BLIND LLM ACCURACY TEST  (15 Cases)  [lora-v2 @ :8000]")
    print("=" * 70)

    results = []
    incorrect_reasoning = []

    for case in OPTION_A_CASES:
        cid = case["id"]
        payload = build_payload(
            case["fico"], case["dti"], case["defaults"],
            case["income"], case["purpose"], cid
        )
        print(f"\n  > Case {cid:02d}  FICO={case['fico']} DTI={case['dti']}%  Defaults={case['defaults']}", end="", flush=True)

        resp = call_evaluate(payload)

        if "error" in resp and "agent_decision" not in resp:
            print(f"  [ERROR: {resp['error']}]")
            results.append({
                "id": cid, "fico": case["fico"], "dti": case["dti"],
                "defaults": case["defaults"],
                "exp_risk": case["exp_risk"], "exp_decision": case["exp_decision"],
                "llm_risk": "ERROR", "llm_decision": "ERROR",
                "llm_rate": "ERROR", "verdict": "INCORRECT", "points": 0.0,
                "reasoning": f"API error: {resp.get('error', 'unknown')}"
            })
            continue

        agent = resp.get("agent_decision", {})
        llm_risk = agent.get("risk_level", "N/A")
        llm_decision = agent.get("loan_decision", "N/A")
        llm_rate = agent.get("interest_rate_tier", "N/A")
        reasoning = agent.get("reasoning", "") or resp.get("reasoning", "")

        verdict, points = score_case(llm_risk, llm_decision, case["exp_risk"], case["exp_decision"])
        print(f"  -> LLM: {llm_risk}/{llm_decision}  Expected: {case['exp_risk']}/{case['exp_decision']}  [{verdict}]")

        results.append({
            "id": cid, "fico": case["fico"], "dti": case["dti"],
            "defaults": case["defaults"],
            "exp_risk": case["exp_risk"], "exp_decision": case["exp_decision"],
            "llm_risk": llm_risk, "llm_decision": llm_decision,
            "llm_rate": llm_rate, "verdict": verdict, "points": points,
            "reasoning": reasoning
        })

        if verdict == "INCORRECT":
            incorrect_reasoning.append({"case_id": cid, "reasoning": reasoning, "result": results[-1]})

    return results, incorrect_reasoning


def run_option_b():
    print("\n" + "=" * 70)
    print("  OPTION B — GRADER vs LLM DISAGREEMENT TEST  (10 Edge Cases)  [lora-v2]")
    print("=" * 70)

    results = []

    for case in OPTION_B_CASES:
        cid = case["id"]
        print(f"\n  > Edge {cid}  FICO={case['fico']} DTI={case['dti']}%  Defaults={case['defaults']}  ({case['note']})", end="", flush=True)

        profile = build_profile_for_grader(case["fico"], case["dti"], case["defaults"], case["income"], cid)
        gt = calculate_dynamic_ground_truth(profile)
        grader_risk = gt.risk_level.value
        grader_decision = gt.loan_decision.value
        grader_rate = gt.interest_rate_tier.value

        payload = build_payload(case["fico"], case["dti"], case["defaults"], case["income"], "general", cid)
        resp = call_evaluate(payload)

        if "error" in resp and "agent_decision" not in resp:
            llm_risk = llm_decision = llm_rate = llm_reasoning = "ERROR"
        else:
            agent = resp.get("agent_decision", {})
            llm_risk = agent.get("risk_level", "N/A")
            llm_decision = agent.get("loan_decision", "N/A")
            llm_rate = agent.get("interest_rate_tier", "N/A")
            llm_reasoning = agent.get("reasoning", "") or resp.get("reasoning", "")

        risk_agree = normalize_risk(llm_risk) == grader_risk
        dec_agree = normalize_decision(llm_decision) == grader_decision

        if risk_agree and dec_agree:
            agreement = "AGREE"
        elif risk_agree or dec_agree:
            agreement = "PARTIAL"
        else:
            agreement = "DISAGREE"

        print(f"\n    Grader: {grader_risk}/{grader_decision}/{grader_rate}")
        print(f"    LLM:    {llm_risk}/{llm_decision}/{llm_rate}  [{agreement}]")

        results.append({
            "id": cid,
            "fico": case["fico"], "dti": case["dti"], "defaults": case["defaults"],
            "note": case["note"],
            "grader_risk": grader_risk, "grader_decision": grader_decision, "grader_rate": grader_rate,
            "llm_risk": llm_risk, "llm_decision": llm_decision, "llm_rate": llm_rate,
            "agreement": agreement,
            "llm_reasoning": llm_reasoning,
        })

    return results


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("  LOAN UNDERWRITING LLM INTELLIGENCE TEST v3 — lora-v2 — Apr 2026")
    print("#" * 70)

    try:
        h = requests.get(f"{BASE_URL}/health", timeout=10)
        hj = h.json()
        print(f"\n  Server: {hj.get('status', 'unknown')} — model env: {hj.get('env_vars', {})}")
    except Exception as e:
        print(f"\n  Server not reachable at {BASE_URL}: {e}")
        sys.exit(1)

    a_results, a_incorrect = run_option_a()
    b_results = run_option_b()

    output = {
        "option_a": a_results,
        "option_b": b_results,
        "incorrect_reasoning": a_incorrect,
    }
    with open("llm_test_results_v3.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved -> llm_test_results_v3.json")
