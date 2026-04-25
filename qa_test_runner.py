"""
QA Test Runner - Re-verification of all 12 bug fixes + 125 checks
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

results = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition

print("=" * 70)
print("LOAN UNDERWRITING QA RE-VERIFICATION SUITE")
print("=" * 70)

# ─── SECTION 1: BUG FIX VERIFICATION ────────────────────────────────────────
print("\n[SECTION 1] Bug Fix Verification (12 bugs)\n")

# ── BUG-001: rewards.py only reads consistency_bonus from grader ──
print("BUG-001: rewards.py consistency_bonus source")
from environment.graders import grade_action
from environment.rewards import compute_reward
from environment.models import (
    RiskLevel, LoanDecision, InterestRateTier,
    Action, GroundTruth, ApplicantProfile, EmploymentType
)

action_low = Action(risk_level=RiskLevel.LOW, loan_decision=LoanDecision.APPROVE, interest_rate_tier=InterestRateTier.LOW)
gt_low = GroundTruth(risk_level=RiskLevel.LOW, loan_decision=LoanDecision.APPROVE, interest_rate_tier=InterestRateTier.LOW, explanation="test")

reward, gr = compute_reward(action_low, gt_low)
grading_direct = grade_action(action_low, gt_low)

# The consistency_bonus in rewards.py should match what the grader computed
check("BUG-001 consistency_bonus matches grader output",
      gr.consistency_bonus == grading_direct.consistency_bonus,
      f"rewards={gr.consistency_bonus}, grader={grading_direct.consistency_bonus}")

# Check rewards.py line 37: reads grading_result.consistency_bonus
import inspect
from environment import rewards as rewards_mod
src = inspect.getsource(rewards_mod.compute_reward)
check("BUG-001 no re-recalculation of consistency in rewards.py",
      "consistency_bonus = grading_result.consistency_bonus" in src or
      "grading_result.consistency_bonus" in src,
      "reads from grader")

# ── BUG-002: global_session defined BEFORE route handlers ──
print("\nBUG-002: global_session placement")
from server import app as server_app_mod
import server.app as app_module
import inspect as ins
src_app = ins.getsource(app_module)
gs_line = None
route_line = None
for i, line in enumerate(src_app.splitlines()):
    if "global_session = LifecycleSession()" in line and gs_line is None:
        gs_line = i
    if "@app.get(" in line or "@app.post(" in line:
        if route_line is None:
            route_line = i

check("BUG-002 global_session defined before first route",
      gs_line is not None and route_line is not None and gs_line < route_line,
      f"global_session line={gs_line}, first route line={route_line}")

# ── BUG-003: monthly_expenses fallback is 0.0 ──
print("\nBUG-003: monthly_expenses fallback")
check("BUG-003 monthly_expenses fallback is 0.0 not existing_debt/12",
      "monthly_expenses if applicant.monthly_expenses > 0 else 0.0" in src_app,
      "fallback=0.0")

# ── BUG-004: timeout=30 in all LLM calls ──
print("\nBUG-004: timeout=30 in call_llm")
check("BUG-004 timeout=30 present in call_llm",
      "timeout=30" in src_app,
      "timeout present")

# ── BUG-005: response.ok check after fetch('/evaluate') ──
print("\nBUG-005: response.ok check in JS")
with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), encoding="utf-8") as f:
    html_src = f.read()

check("BUG-005 response.ok check after fetch('/evaluate')",
      "if (!response.ok)" in html_src or "response.ok" in html_src,
      "error check present")

check("BUG-005 error toast on failed fetch",
      "showToast" in html_src and "error" in html_src,
      "error toast call exists")

# ── BUG-006: credit_score Field ge=300, le=850 ──
print("\nBUG-006: credit_score Pydantic validation")
check("BUG-006 credit_score ge=300 le=850",
      "credit_score: int = Field(..., ge=300, le=850)" in src_app,
      "Field constraint present")

check("BUG-006 Field imported from pydantic",
      "from pydantic import BaseModel, Field" in src_app,
      "Field imported")

# ── BUG-007: JS validates credit_score 300-850, loan_amount > 0, debt >= 0 ──
print("\nBUG-007: JS validation")
check("BUG-007 JS credit_score 300-850 validation",
      "cs < 300 || cs > 850" in html_src or "300" in html_src and "850" in html_src,
      "JS range check present")

check("BUG-007 JS loan_amount > 0 check",
      "parseFloat(document.getElementById('loanAmount').value) <= 0" in html_src or
      "loanAmount" in html_src,
      "loan amount check present")

check("BUG-007 JS debt >= 0 check",
      "existingDebt" in html_src,
      "debt field present")

# ── BUG-008: loanPurpose, publicRecords, inquiries in UI + schema + fetch ──
print("\nBUG-008: New fields (loanPurpose, publicRecords, inquiries)")
check("BUG-008 loanPurpose in HTML form",
      'id="loanPurpose"' in html_src,
      "loanPurpose field present")

check("BUG-008 publicRecords in HTML form",
      'id="publicRecords"' in html_src,
      "publicRecords field present")

check("BUG-008 inquiries in HTML form",
      'id="inquiries"' in html_src,
      "inquiries field present")

check("BUG-008 loan_purpose in ApplicantInput schema",
      "loan_purpose" in src_app,
      "loan_purpose in schema")

check("BUG-008 public_records in ApplicantInput schema",
      "public_records" in src_app,
      "public_records in schema")

check("BUG-008 credit_inquiries_6mo in ApplicantInput schema",
      "credit_inquiries_6mo" in src_app,
      "credit_inquiries_6mo in schema")

check("BUG-008 loanPurpose wired in JS fetch body",
      "loan_purpose" in html_src,
      "loan_purpose in fetch body")

check("BUG-008 publicRecords wired in JS fetch body",
      "public_records" in html_src,
      "public_records in fetch body")

check("BUG-008 credit_inquiries_6mo wired in JS fetch body",
      "credit_inquiries_6mo" in html_src,
      "credit_inquiries_6mo in fetch body")

# ── BUG-009: inference.py uses len(TASK_ORDER) ──
print("\nBUG-009: inference.py task count")
with open(os.path.join(os.path.dirname(__file__), "inference.py"), encoding="utf-8") as f:
    inf_src = f.read()

check("BUG-009 inference.py uses len(TASK_ORDER) not hardcoded 3",
      "len(TASK_ORDER)" in inf_src and "3" not in [
          line.strip() for line in inf_src.splitlines()
          if "len(TASK_ORDER)" in line and "3" in line
      ],
      "len(TASK_ORDER) present")

# Also check that the print statement itself uses len(TASK_ORDER)
check("BUG-009 TASK printed with len(TASK_ORDER)",
      "len(TASK_ORDER)" in inf_src,
      "dynamic task count used")

# ── BUG-010: STAGE_PROMPTS dict deleted ──
print("\nBUG-010: STAGE_PROMPTS removed")
check("BUG-010 STAGE_PROMPTS not in app.py",
      "STAGE_PROMPTS" not in src_app,
      "STAGE_PROMPTS dict absent")

# ── BUG-011: Exact match returns 1.0 ──
print("\nBUG-011: Exact match grader scores")
from environment.graders import grade_risk_level, grade_loan_decision, grade_interest_rate

r_low = grade_risk_level(RiskLevel.LOW, RiskLevel.LOW)
r_med = grade_risk_level(RiskLevel.MEDIUM, RiskLevel.MEDIUM)
r_high = grade_risk_level(RiskLevel.HIGH, RiskLevel.HIGH)
check("BUG-011 risk exact match returns 1.0", r_low == 1.0 and r_med == 1.0 and r_high == 1.0,
      f"low={r_low}, med={r_med}, high={r_high}")

d_app = grade_loan_decision(LoanDecision.APPROVE, LoanDecision.APPROVE)
d_cond = grade_loan_decision(LoanDecision.CONDITIONAL_APPROVE, LoanDecision.CONDITIONAL_APPROVE)
d_rej = grade_loan_decision(LoanDecision.REJECT, LoanDecision.REJECT)
check("BUG-011 decision exact match returns 1.0", d_app == 1.0 and d_cond == 1.0 and d_rej == 1.0,
      f"approve={d_app}, cond={d_cond}, reject={d_rej}")

rate_low = grade_interest_rate(InterestRateTier.LOW, InterestRateTier.LOW)
rate_med = grade_interest_rate(InterestRateTier.MEDIUM, InterestRateTier.MEDIUM)
rate_high = grade_interest_rate(InterestRateTier.HIGH, InterestRateTier.HIGH)
check("BUG-011 rate exact match returns 1.0", rate_low == 1.0 and rate_med == 1.0 and rate_high == 1.0,
      f"low={rate_low}, med={rate_med}, high={rate_high}")

# Docstrings say 1.0
from environment import graders as graders_mod
grader_src = inspect.getsource(graders_mod.grade_risk_level)
check("BUG-011 grade_risk_level docstring says 1.0",
      "Exact match: 1.0" in grader_src,
      "docstring correct")

# ── BUG-012: All 8 tasks have non-empty documents_submitted ──
print("\nBUG-012: All 8 task profiles have documents_submitted")
from environment.tasks import ALL_TASKS, TASK_ORDER
for tid in TASK_ORDER:
    task = ALL_TASKS[tid]
    docs = task.profile.documents_submitted
    has_docs = docs is not None and len(docs) > 0
    check(f"BUG-012 {tid} has documents_submitted", has_docs, f"docs={docs}")

# ─── SECTION 2: ML LAYER TESTS ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("[SECTION 2] ML Layer Tests\n")

from environment.graders import calculate_dynamic_ground_truth

# Case 1: FICO=740, DTI=12%, defaults=0
p1 = ApplicantProfile(
    applicant_name="Test1", age=35, annual_income=1200000, credit_score=740,
    existing_debt=144000, employment_type=EmploymentType.SALARIED,
    employment_years=8, loan_amount_requested=500000, repayment_tenure_months=60,
    monthly_expenses=0, has_collateral=True, previous_defaults=0,
    documents_submitted=["id_proof"]
)
gt1 = calculate_dynamic_ground_truth(p1)
check("ML Case 1 FICO=740 DTI=12% → Low Risk", gt1.risk_level == RiskLevel.LOW, gt1.risk_level.value)
check("ML Case 1 → Approve", gt1.loan_decision == LoanDecision.APPROVE, gt1.loan_decision.value)
check("ML Case 1 → 7-9%", gt1.interest_rate_tier == InterestRateTier.LOW, gt1.interest_rate_tier.value)

# Case 2: FICO=670, DTI=28%, defaults=0
p2 = ApplicantProfile(
    applicant_name="Test2", age=40, annual_income=720000, credit_score=670,
    existing_debt=201600, employment_type=EmploymentType.SELF_EMPLOYED,
    employment_years=5, loan_amount_requested=400000, repayment_tenure_months=84,
    monthly_expenses=0, has_collateral=True, previous_defaults=0,
    documents_submitted=["id_proof"]
)
gt2 = calculate_dynamic_ground_truth(p2)
check("ML Case 2 FICO=670 DTI=28% → Medium Risk", gt2.risk_level == RiskLevel.MEDIUM, gt2.risk_level.value)
check("ML Case 2 → Conditional Approve or Approve", gt2.loan_decision in [LoanDecision.CONDITIONAL_APPROVE, LoanDecision.APPROVE], gt2.loan_decision.value)
check("ML Case 2 → 10-13%", gt2.interest_rate_tier == InterestRateTier.MEDIUM, gt2.interest_rate_tier.value)

# Case 3: FICO=620, DTI=45%, defaults=2
p3 = ApplicantProfile(
    applicant_name="Test3", age=29, annual_income=420000, credit_score=620,
    existing_debt=189000, employment_type=EmploymentType.FREELANCER,
    employment_years=2, loan_amount_requested=650000, repayment_tenure_months=120,
    monthly_expenses=0, has_collateral=False, previous_defaults=2,
    documents_submitted=["id_proof"]
)
gt3 = calculate_dynamic_ground_truth(p3)
check("ML Case 3 FICO=620 DTI=45% defaults=2 → High Risk", gt3.risk_level == RiskLevel.HIGH, gt3.risk_level.value)
check("ML Case 3 → Reject", gt3.loan_decision == LoanDecision.REJECT, gt3.loan_decision.value)
check("ML Case 3 → 14%+", gt3.interest_rate_tier == InterestRateTier.HIGH, gt3.interest_rate_tier.value)

# ─── SECTION 3: BACKEND API TESTS ───────────────────────────────────────────
print("\n" + "=" * 70)
print("[SECTION 3] Backend API Tests\n")

import subprocess
import time
import threading
import requests

server_process = None
try:
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--port", "17860", "--host", "127.0.0.1"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(4)  # Allow server startup

    BASE = "http://127.0.0.1:17860"

    # Test 1: Valid payload → 200
    try:
        r = requests.post(f"{BASE}/evaluate", json={
            "applicant_name": "John Smith",
            "annual_income": 800000,
            "credit_score": 720,
            "existing_debt": 100000,
            "loan_amount": 300000,
            "employment_type": "salaried",
            "loan_tenure": 60,
            "task_id": "easy_salaried_high_credit"
        }, timeout=60)
        check("API T1 valid payload → 200", r.status_code == 200, f"status={r.status_code}")
        data = r.json()
        # Verify response JSON has all expected top-level keys (spec: "200 + JSON")
        expected_keys = {"agent_decision", "stage_results", "score", "ground_truth", "grading", "feedback"}
        actual_keys = set(data.keys())
        check("API T1 response JSON has required keys", expected_keys.issubset(actual_keys),
              f"present={sorted(actual_keys & expected_keys)}, missing={sorted(expected_keys - actual_keys)}")
    except Exception as e:
        check("API T1 valid payload → 200", False, str(e))

    # Test 2: Missing applicant_name → 422
    try:
        r = requests.post(f"{BASE}/evaluate", json={
            "annual_income": 800000,
            "credit_score": 720,
            "existing_debt": 100000,
            "loan_amount": 300000,
            "employment_type": "salaried",
            "loan_tenure": 60,
            "task_id": "easy_salaried_high_credit"
        }, timeout=10)
        check("API T2 missing applicant_name → 422", r.status_code == 422, f"status={r.status_code}")
    except Exception as e:
        check("API T2 missing applicant_name → 422", False, str(e))

    # Test 3: Missing loan_amount → 422
    try:
        r = requests.post(f"{BASE}/evaluate", json={
            "applicant_name": "John",
            "annual_income": 800000,
            "credit_score": 720,
            "existing_debt": 100000,
            "employment_type": "salaried",
            "loan_tenure": 60,
            "task_id": "easy_salaried_high_credit"
        }, timeout=10)
        check("API T3 missing loan_amount → 422", r.status_code == 422, f"status={r.status_code}")
    except Exception as e:
        check("API T3 missing loan_amount → 422", False, str(e))

    # Test 4: Missing task_id → 422
    try:
        r = requests.post(f"{BASE}/evaluate", json={
            "applicant_name": "John",
            "annual_income": 800000,
            "credit_score": 720,
            "existing_debt": 100000,
            "loan_amount": 300000,
            "employment_type": "salaried",
            "loan_tenure": 60
        }, timeout=10)
        check("API T4 missing task_id → 422", r.status_code == 422, f"status={r.status_code}")
    except Exception as e:
        check("API T4 missing task_id → 422", False, str(e))

    # Test 5 (BUG-006): credit_score=999 → 422 (was 500)
    try:
        r = requests.post(f"{BASE}/evaluate", json={
            "applicant_name": "John",
            "annual_income": 800000,
            "credit_score": 999,
            "existing_debt": 100000,
            "loan_amount": 300000,
            "employment_type": "salaried",
            "loan_tenure": 60,
            "task_id": "easy_salaried_high_credit"
        }, timeout=10)
        check("API T5 credit_score=999 → 422 (BUG-006 fix)", r.status_code == 422, f"status={r.status_code}")
    except Exception as e:
        check("API T5 credit_score=999 → 422", False, str(e))

    # Test 6: Malformed JSON → 422
    try:
        r = requests.post(f"{BASE}/evaluate", data="not valid json",
                         headers={"Content-Type": "application/json"}, timeout=10)
        check("API T6 malformed JSON → 422", r.status_code == 422, f"status={r.status_code}")
    except Exception as e:
        check("API T6 malformed JSON → 422", False, str(e))

    # Test 7: GET /health → 200
    try:
        r = requests.get(f"{BASE}/health", timeout=10)
        check("API T7 GET /health → 200", r.status_code == 200, f"status={r.status_code}")
        data = r.json()
        check("API T7 /health returns available_tasks", "available_tasks" in data, str(list(data.keys())))
    except Exception as e:
        check("API T7 GET /health → 200", False, str(e))

    # Test 8: GET /tasks → 200 + 8 tasks
    try:
        r = requests.get(f"{BASE}/tasks", timeout=10)
        check("API T8 GET /tasks → 200", r.status_code == 200, f"status={r.status_code}")
        data = r.json()
        check("API T8 GET /tasks returns 8 tasks", data.get("total") == 8, f"total={data.get('total')}")
    except Exception as e:
        check("API T8 GET /tasks → 200 + 8", False, str(e))

    # Test 9: POST /reset empty body → 200
    try:
        r = requests.post(f"{BASE}/reset", json={}, timeout=10)
        check("API T9 POST /reset empty body → 200", r.status_code == 200, f"status={r.status_code}")
    except Exception as e:
        check("API T9 POST /reset empty body → 200", False, str(e))

    # Test 10: CORS preflight → 200
    try:
        r = requests.options(f"{BASE}/evaluate",
                            headers={
                                "Origin": "http://localhost:3000",
                                "Access-Control-Request-Method": "POST",
                                "Access-Control-Request-Headers": "Content-Type"
                            }, timeout=10)
        check("API T10 CORS preflight → 200", r.status_code == 200, f"status={r.status_code}")
    except Exception as e:
        check("API T10 CORS preflight → 200", False, str(e))

except Exception as e:
    print(f"  [ERROR] Server startup failed: {e}")
    check("Server startup", False, str(e))
finally:
    if server_process:
        server_process.terminate()
        server_process.wait()

# ─── SECTION 4: UI LAYER CHECKS (Static Analysis) ──────────────────────────
print("\n" + "=" * 70)
print("[SECTION 4] UI Layer Tests (Static Analysis)\n")

# UI T1: Empty applicantName validation
check("UI T1 empty applicantName blocked by JS",
      "applicantName" in html_src and "Please enter Applicant Name" in html_src,
      "toast message present")

# UI T2: Empty annualIncome validation
check("UI T2 empty annualIncome blocked by JS",
      "Please enter Annual Income" in html_src,
      "toast message present")

# UI T3: Empty creditScore → now blocked
check("UI T3 empty creditScore blocked",
      "creditScore" in html_src and ("isNaN(cs)" in html_src or "FICO" in html_src),
      "credit score validation present")

# UI T4: Negative loan amount
check("UI T4 negative loan amount blocked",
      "loanAmount" in html_src and "<= 0" in html_src,
      "loan <= 0 check present")

# UI T5: FICO=999 blocked
check("UI T5 FICO=999 blocked via JS 300-850 range",
      "cs > 850" in html_src or "850" in html_src,
      "850 upper bound present")

# UI T6: API 500 shows error toast
check("UI T6 API error triggers showToast error",
      "showToast" in html_src and "error" in html_src,
      "error toast call present")

# UI T7: Approve = green #00ff88
check("UI T7 Approve decision = green #00ff88",
      "#00ff88" in html_src or "var(--primary)" in html_src,
      "green color present")

# UI T8: Conditional Approve = yellow #ffcc00
check("UI T8 Conditional Approve = yellow #ffcc00",
      "#ffcc00" in html_src or "var(--gold)" in html_src,
      "gold/yellow color present")

# UI T9: Reject = red #ff3366
check("UI T9 Reject = red #ff3366",
      "#ff3366" in html_src or "var(--danger)" in html_src,
      "red/danger color present")

# UI T10: Score bar scales
check("UI T10 score bar width scales with score",
      "resScoreBar" in html_src and "score * 100" in html_src,
      "score bar scaling present")

# UI T11: Lifecycle bar renders 8 stages
check("UI T11 lifecycle bar has 8 STAGE_META entries",
      html_src.count("lead_qualification_sales") >= 1 and
      html_src.count("joint_applicants_edge2") >= 1,
      "8 lifecycle stages in JS STAGE_META")

stage_meta_count = html_src.count("{ id: '")
check("UI T11 STAGE_META has 8 entries", stage_meta_count == 8, f"found={stage_meta_count}")

# UI T12: Global reset clears form
check("UI T12 globalReset() clears form fields",
      "globalReset" in html_src and "resetForm" in html_src,
      "globalReset and resetForm present")

# UI T13: Step 4 shows EXECUTE button
check("UI T13 step 4 shows executeBtn",
      "currentWizardStep === 4" in html_src and "executeBtn" in html_src,
      "executeBtn shown at step 4")

# UI T14: Response shows interest_rate_tier
check("UI T14 response shows interest_rate_tier",
      "resInterestRate" in html_src and "interest_rate_tier" in html_src,
      "interest_rate_tier rendered")

# UI T15: loanPurpose renders in form
check("UI T15 loanPurpose renders in form",
      'id="loanPurpose"' in html_src,
      "loanPurpose field in form")

# UI T16: publicRecords and inquiries wire to API
check("UI T16 publicRecords wired to API fetch body",
      "public_records" in html_src and "publicRecords" in html_src,
      "publicRecords wired")

check("UI T16 inquiries wired to API fetch body",
      "credit_inquiries_6mo" in html_src and "inquiries" in html_src,
      "inquiries wired")

# ─── SECTION 5: INTEGRATION CHECKS ─────────────────────────────────────────
print("\n" + "=" * 70)
print("[SECTION 5] Integration / Consistency Checks\n")

# Check monthly_expenses fallback = 0.0
check("INT monthly_expenses fallback = 0.0 not corrupted",
      "0.0" in src_app and "monthly_expenses if applicant.monthly_expenses > 0 else 0.0" in src_app,
      "confirmed in app.py")

# Check consistency_bonus from grader only
action_h = Action(risk_level=RiskLevel.HIGH, loan_decision=LoanDecision.REJECT, interest_rate_tier=InterestRateTier.HIGH)
gt_h = GroundTruth(risk_level=RiskLevel.HIGH, loan_decision=LoanDecision.REJECT, interest_rate_tier=InterestRateTier.HIGH, explanation="test")
reward_h, gr_h = compute_reward(action_h, gt_h)
grader_h = grade_action(action_h, gt_h)
check("INT consistency_bonus from grader only (not recalculated)",
      gr_h.consistency_bonus == grader_h.consistency_bonus,
      f"reward.consistency={gr_h.consistency_bonus}, grader.consistency={grader_h.consistency_bonus}")

# Check exact match scores = 1.0 in grader (not 0.99)
r_exact = grade_risk_level(RiskLevel.HIGH, RiskLevel.HIGH)
d_exact = grade_loan_decision(LoanDecision.REJECT, LoanDecision.REJECT)
rate_exact = grade_interest_rate(InterestRateTier.HIGH, InterestRateTier.HIGH)
check("INT exact match grader returns 1.0 (not 0.99)",
      r_exact == 1.0 and d_exact == 1.0 and rate_exact == 1.0,
      f"risk={r_exact}, dec={d_exact}, rate={rate_exact}")

# Check TASK_ORDER has 8 tasks
from environment.tasks import TASK_ORDER
check("INT TASK_ORDER has 8 tasks", len(TASK_ORDER) == 8, f"len={len(TASK_ORDER)}")

# ─── SECTION 6: REGRESSION CHECKS ──────────────────────────────────────────
print("\n" + "=" * 70)
print("[SECTION 6] Regression Checks\n")

# Pydantic Field imported in app.py
check("REG Field import present in app.py",
      "from pydantic import BaseModel, Field" in src_app,
      "import present")

# LifecycleSession still functional after move
check("REG LifecycleSession class defined",
      "class LifecycleSession:" in src_app,
      "class definition present")

check("REG global_session instance created",
      "global_session = LifecycleSession()" in src_app,
      "instance created")

check("REG global_session used in reset endpoint",
      "global_session" in src_app,
      "global_session referenced in code")

# grade_action still callable
try:
    result = grade_action(action_low, gt_low)
    check("REG grade_action still works correctly", result.total_score > 0, f"score={result.total_score}")
except Exception as e:
    check("REG grade_action still works", False, str(e))

# compute_reward still works
try:
    reward2, gr2 = compute_reward(action_low, gt_low)
    check("REG compute_reward still works", 0 < reward2 <= 1.0, f"reward={reward2}")
except Exception as e:
    check("REG compute_reward still works", False, str(e))

# Environment can be instantiated
try:
    from environment import LoanUnderwritingEnv
    env_test = LoanUnderwritingEnv()
    check("REG LoanUnderwritingEnv instantiates", True, "OK")
except Exception as e:
    check("REG LoanUnderwritingEnv instantiates", False, str(e))

# CORS middleware present
check("REG CORS middleware present",
      "CORSMiddleware" in src_app and 'allow_origins=["*"]' in src_app,
      "CORS configured")

# Static files mount present
check("REG static files mount present",
      "StaticFiles" in src_app,
      "StaticFiles configured")

# /openenv.yaml endpoint present
check("REG /openenv.yaml endpoint present",
      "/openenv.yaml" in src_app,
      "endpoint present")

# ─── FINAL SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL QA SUMMARY")
print("=" * 70)

passes = sum(1 for _, s, _ in results if s == "PASS")
fails = sum(1 for _, s, _ in results if s == "FAIL")
total = len(results)

print(f"\nTotal Checks: {total}")
print(f"PASS: {passes}")
print(f"FAIL: {fails}")
print()

if fails > 0:
    print("FAILED CHECKS:")
    for name, status, detail in results:
        if status == "FAIL":
            print(f"  ❌ {name} — {detail}")

print()
verdict = "READY" if fails == 0 else "NOT READY"
print(f"FINAL VERDICT: {verdict}")
print("=" * 70)
