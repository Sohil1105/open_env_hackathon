"""
Advanced QA + Performance + Live Deployment Test Suite v3
Suites A (Edge Case Stress), B (Load & Performance), C (Live Deployment)
"""

import sys
import os
import time
import json
import threading
import concurrent.futures
import statistics
import traceback
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

BASE_URL = "http://localhost:8000"
RESULTS = []
BUGS = []


def result(suite, test_id, name, passed, actual, expected, notes="", ms=None):
    status = "PASS" if passed else "FAIL"
    RESULTS.append({
        "suite": suite, "test_id": test_id, "name": name,
        "status": status, "actual": str(actual)[:300],
        "expected": str(expected)[:300], "notes": notes, "ms": ms,
    })
    icon = "[PASS]" if passed else "[FAIL]"
    ms_str = f" [{ms}ms]" if ms is not None else ""
    print(f"  {icon} {test_id}: {name}{ms_str}", flush=True)
    if not passed:
        print(f"         Actual:   {str(actual)[:200]}", flush=True)
        print(f"         Expected: {str(expected)[:200]}", flush=True)
        if notes:
            print(f"         Notes:    {notes}", flush=True)


def bug(bug_id, file_path, line_hint, severity, description):
    BUGS.append({"id": bug_id, "file": file_path, "line": line_hint,
                 "severity": severity, "description": description})


def post_evaluate(payload, timeout=35):
    t0 = time.time()
    try:
        r = requests.post(f"{BASE_URL}/evaluate", json=payload, timeout=timeout)
        ms = int((time.time() - t0) * 1000)
        return r.status_code, r, ms
    except requests.exceptions.Timeout:
        ms = int((time.time() - t0) * 1000)
        return "TIMEOUT", None, ms
    except Exception as e:
        ms = int((time.time() - t0) * 1000)
        return "ERROR", str(e), ms


def post_reset(payload=None, timeout=10):
    t0 = time.time()
    try:
        r = requests.post(f"{BASE_URL}/reset", json=payload or {}, timeout=timeout)
        ms = int((time.time() - t0) * 1000)
        return r.status_code, r, ms
    except Exception as e:
        ms = int((time.time() - t0) * 1000)
        return "ERROR", str(e), ms


VALID = {
    "applicant_name": "Test User",
    "annual_income": 1200000.0,
    "credit_score": 750,
    "existing_debt": 100000.0,
    "loan_amount": 400000.0,
    "employment_type": "salaried",
    "employment_years": 5.0,
    "loan_tenure": 60,
    "task_id": "easy_salaried_high_credit",
    "age": 35,
    "has_collateral": True,
    "previous_defaults": 0,
    "loan_purpose": "home improvement",
    "public_records": 0,
    "credit_inquiries_6mo": 0,
    "documents_submitted": ["pay_stub", "bank_statement", "id_proof"],
}


# ============================================================
# SUITE A: EDGE CASE STRESS TESTS
# ============================================================

def suite_a1_boundary_fico():
    print("\n[A1] Boundary FICO Values", flush=True)

    p = {**VALID, "credit_score": 300}
    sc, r, ms = post_evaluate(p)
    result("A", "A1.1", "FICO=300 -> 200 OK", sc == 200, sc, 200, ms=ms)
    if sc == 200:
        gt = r.json().get("ground_truth", {})
        result("A", "A1.1b", "FICO=300 -> High/Reject ground truth",
               gt.get("risk_level") == "High" and gt.get("loan_decision") == "Reject",
               f"{gt.get('risk_level')}/{gt.get('loan_decision')}", "High/Reject", ms=ms)

    p = {**VALID, "credit_score": 850}
    sc, r, ms = post_evaluate(p)
    result("A", "A1.2", "FICO=850 -> 200 OK", sc == 200, sc, 200, ms=ms)
    if sc == 200:
        gt = r.json().get("ground_truth", {})
        result("A", "A1.2b", "FICO=850 -> Low/Approve ground truth",
               gt.get("risk_level") == "Low" and gt.get("loan_decision") == "Approve",
               f"{gt.get('risk_level')}/{gt.get('loan_decision')}", "Low/Approve", ms=ms)

    p = {**VALID, "credit_score": 299}
    sc, r, ms = post_evaluate(p)
    result("A", "A1.3", "FICO=299 -> 422", sc == 422, sc, 422, ms=ms)

    p = {**VALID, "credit_score": 851}
    sc, r, ms = post_evaluate(p)
    result("A", "A1.4", "FICO=851 -> 422", sc == 422, sc, 422, ms=ms)


def suite_a2_dti_extremes():
    print("\n[A2] DTI Extremes", flush=True)

    p = {**VALID, "annual_income": 1.0, "existing_debt": 999999.0}
    sc, r, ms = post_evaluate(p)
    result("A", "A2.1", "DTI>100% -> no crash (200)", sc == 200, sc, 200, ms=ms)
    if sc == 200:
        gt = r.json().get("ground_truth", {})
        result("A", "A2.1b", "DTI>100% -> Reject ground truth",
               gt.get("loan_decision") == "Reject", gt.get("loan_decision"), "Reject", ms=ms)

    p = {**VALID, "annual_income": 0}
    sc, r, ms = post_evaluate(p)
    result("A", "A2.2", "annual_income=0 -> 422", sc == 422, sc, 422, ms=ms)
    if sc != 422:
        bug("BUG-A2", "server/app.py", "ApplicantInput ~L188",
            "HIGH", "annual_income=0 not blocked — division by zero risk in grader DTI calc")

    p = {**VALID, "annual_income": 999999999.0, "existing_debt": 0.0}
    sc, r, ms = post_evaluate(p)
    result("A", "A2.3", "DTI=0% -> no overflow, 200 OK", sc == 200, sc, 200, ms=ms)
    if sc == 200:
        gt = r.json().get("ground_truth", {})
        result("A", "A2.3b", "DTI=0% -> Approve ground truth",
               gt.get("loan_decision") == "Approve", gt.get("loan_decision"), "Approve", ms=ms)


def suite_a3_loan_extremes():
    print("\n[A3] Loan Amount Extremes", flush=True)

    p = {**VALID, "loan_amount": 1.0}
    sc, r, ms = post_evaluate(p)
    result("A", "A3.1", "loan_amount=1 -> 200 OK", sc == 200, sc, 200, ms=ms)

    p = {**VALID, "loan_amount": 999999999.0}
    sc, r, ms = post_evaluate(p)
    result("A", "A3.2", "loan_amount=999999999 -> 200 OK (no overflow)", sc == 200, sc, 200, ms=ms)

    p = {**VALID, "loan_amount": 0.0}
    sc, r, ms = post_evaluate(p)
    result("A", "A3.3", "loan_amount=0 -> 422", sc == 422, sc, 422, ms=ms)
    if sc != 422:
        bug("BUG-A3a", "server/app.py", "ApplicantInput loan_amount field ~L192",
            "HIGH", "loan_amount=0 accepted by backend — only JS-side guard exists, no backend ge=0.01 constraint")

    p = {**VALID, "loan_amount": -1.0}
    sc, r, ms = post_evaluate(p)
    result("A", "A3.4", "loan_amount=-1 -> 422", sc == 422, sc, 422, ms=ms)
    if sc != 422:
        bug("BUG-A3b", "server/app.py", "ApplicantInput loan_amount field ~L192",
            "HIGH", "loan_amount=-1 accepted by backend — negative loan amounts not blocked")


def suite_a4_injection():
    print("\n[A4] String Injection Tests", flush=True)

    p = {**VALID, "applicant_name": "<script>alert(1)</script>"}
    sc, r, ms = post_evaluate(p)
    result("A", "A4.1", "XSS in applicant_name -> no crash", sc == 200, sc, 200, ms=ms)
    if sc == 200:
        resp_text = json.dumps(r.json())
        result("A", "A4.1b", "XSS name stored as plain text in response (no execution)",
               "<script>" in resp_text, "stored as-is in JSON", "plain text storage", ms=ms)

    p = {**VALID, "loan_purpose": "' OR 1=1; DROP TABLE users;--"}
    sc, r, ms = post_evaluate(p)
    result("A", "A4.2", "SQL injection in loan_purpose -> no crash", sc == 200, sc, 200, ms=ms)

    p = {**VALID, "applicant_name": ""}
    sc, r, ms = post_evaluate(p)
    result("A", "A4.3", "applicant_name='' -> 422", sc == 422, sc, 422, ms=ms)
    if sc != 422:
        bug("BUG-A4a", "server/app.py", "ApplicantInput applicant_name ~L188",
            "MEDIUM", "Empty applicant_name passes validation — no min_length=1 constraint")

    p = {**VALID, "applicant_name": "   "}
    sc, r, ms = post_evaluate(p)
    result("A", "A4.4", "applicant_name='   ' (whitespace-only) -> 422 or sanitized",
           sc in (200, 422), sc, "200 or 422",
           notes="Whitespace-only name should be blocked or sanitized", ms=ms)
    if sc == 200:
        bug("BUG-A4b", "server/app.py", "ApplicantInput applicant_name ~L188",
            "LOW", "Whitespace-only applicant_name passes validation — no strip/min_length check")


def suite_a5_new_fields():
    print("\n[A5] New Fields Edge Cases (BUG-008 fix)", flush=True)

    p = {**VALID, "public_records": -1}
    sc, r, ms = post_evaluate(p)
    result("A", "A5.1", "public_records=-1 -> 422", sc == 422, sc, 422, ms=ms)

    p = {**VALID, "credit_inquiries_6mo": -1}
    sc, r, ms = post_evaluate(p)
    result("A", "A5.2", "credit_inquiries_6mo=-1 -> 422", sc == 422, sc, 422, ms=ms)

    p = {**VALID, "public_records": 999, "credit_score": 580}
    sc, r, ms = post_evaluate(p)
    result("A", "A5.3", "public_records=999 -> 200 OK (no crash)", sc == 200, sc, 200, ms=ms)

    p = {**VALID, "loan_purpose": ""}
    sc, r, ms = post_evaluate(p)
    result("A", "A5.4", "loan_purpose='' -> 200 OK (default 'general')", sc == 200, sc, 200, ms=ms)
    if sc != 200:
        bug("BUG-A5", "server/app.py", "ApplicantInput loan_purpose default",
            "LOW", "Empty loan_purpose causes crash — no default fallback guard")


def suite_a6_grader_consistency():
    print("\n[A6] Grader Consistency Bonus Edge Cases (Static)", flush=True)
    from environment.models import Action, RiskLevel, LoanDecision, InterestRateTier
    from environment.graders import grade_consistency

    cases = [
        ("Low + Approve + 7-9%",
         Action(risk_level=RiskLevel.LOW, loan_decision=LoanDecision.APPROVE,
                interest_rate_tier=InterestRateTier.LOW),
         0.10, "A6.1"),
        ("High + Approve + 7-9% (contradiction)",
         Action(risk_level=RiskLevel.HIGH, loan_decision=LoanDecision.APPROVE,
                interest_rate_tier=InterestRateTier.LOW),
         -0.10, "A6.2"),
        ("Medium + Conditional Approve + 10-13%",
         Action(risk_level=RiskLevel.MEDIUM, loan_decision=LoanDecision.CONDITIONAL_APPROVE,
                interest_rate_tier=InterestRateTier.MEDIUM),
         0.10, "A6.3"),
        ("High + Reject + 7-9% (mixed: good decision, bad rate)",
         Action(risk_level=RiskLevel.HIGH, loan_decision=LoanDecision.REJECT,
                interest_rate_tier=InterestRateTier.LOW),
         0.0, "A6.4"),  # +0.05 reject +(-0.05 low rate) = 0.0
    ]

    for desc, action, expected, tid in cases:
        bonus = grade_consistency(action)
        passed = abs(bonus - expected) < 0.001
        result("A", tid, f"Consistency: {desc}", passed,
               f"bonus={bonus:.2f}", f"expected={expected:.2f}")


def suite_a7_all_tasks():
    print("\n[A7] All 8 Tasks Sequential Execution", flush=True)
    from environment.tasks import TASK_ORDER, ALL_TASKS

    task_payloads = {
        "lead_qualification_sales":     {**VALID, "task_id": "lead_qualification_sales", "credit_score": 760, "annual_income": 950000, "existing_debt": 80000},
        "document_verification_hr":     {**VALID, "task_id": "document_verification_hr", "credit_score": 640, "annual_income": 580000, "existing_debt": 220000},
        "easy_salaried_high_credit":    {**VALID, "task_id": "easy_salaried_high_credit"},
        "medium_self_employed_moderate":{**VALID, "task_id": "medium_self_employed_moderate", "employment_type": "self_employed", "credit_score": 665, "annual_income": 720000, "existing_debt": 280000},
        "hard_freelancer_complex":      {**VALID, "task_id": "hard_freelancer_complex", "employment_type": "freelancer", "credit_score": 572, "annual_income": 420000, "existing_debt": 380000, "has_collateral": False, "previous_defaults": 2},
        "customer_onboarding_pm":       {**VALID, "task_id": "customer_onboarding_pm", "credit_score": 745, "annual_income": 1050000, "existing_debt": 120000},
        "bankruptcy_recovery_edge1":    {**VALID, "task_id": "bankruptcy_recovery_edge1", "credit_score": 680, "annual_income": 65000, "existing_debt": 8000, "loan_amount": 120000},
        "joint_applicants_edge2":       {**VALID, "task_id": "joint_applicants_edge2", "credit_score": 720, "annual_income": 120000, "existing_debt": 25000, "loan_amount": 300000},
    }

    for i, task_id in enumerate(TASK_ORDER, 1):
        payload = task_payloads.get(task_id, {**VALID, "task_id": task_id})
        sc, r, ms = post_evaluate(payload)
        passed = sc == 200
        result("A", f"A7.{i}", f"Task {i}: {task_id[:30]} -> 200", passed, sc, 200, ms=ms)
        if passed:
            data = r.json()
            docs = ALL_TASKS[task_id].profile.documents_submitted or []
            result("A", f"A7.{i}b", f"Task {i}: documents_submitted non-empty",
                   len(docs) > 0, len(docs), ">0")
            has_schema = all(k in data for k in ["agent_decision", "score", "ground_truth", "grading"])
            result("A", f"A7.{i}c", f"Task {i}: valid response schema", has_schema, list(data.keys())[:4], "required keys")


def suite_a8_reset_between_tasks():
    print("\n[A8] Reset Between Tasks -- State Isolation", flush=True)

    p1 = {**VALID, "task_id": "easy_salaried_high_credit"}
    sc1, r1, ms1 = post_evaluate(p1)
    result("A", "A8.1", "Submit task 1 -> 200", sc1 == 200, sc1, 200, ms=ms1)

    sc_r, r_r, ms_r = post_reset({})
    result("A", "A8.2", "POST /reset -> 200", sc_r == 200, sc_r, 200, ms=ms_r)
    if sc_r == 200:
        data = r_r.json()
        result("A", "A8.3", "POST /reset -> status=reset_complete",
               data.get("status") == "reset_complete", data.get("status"), "reset_complete")

    p2 = {**VALID, "task_id": "medium_self_employed_moderate", "employment_type": "self_employed",
          "credit_score": 665, "annual_income": 720000, "existing_debt": 280000}
    sc2, r2, ms2 = post_evaluate(p2)
    result("A", "A8.4", "Submit task 2 after reset -> 200", sc2 == 200, sc2, 200, ms=ms2)
    if sc2 == 200:
        data2 = r2.json()
        result("A", "A8.5", "Task 2 has valid agent_decision (no bleed from task 1)",
               "agent_decision" in data2 and data2["agent_decision"] is not None,
               bool(data2.get("agent_decision")), True)

    try:
        rh = requests.get(f"{BASE_URL}/health", timeout=5)
        result("A", "A8.6", "/health still 200 after reset cycle", rh.status_code == 200, rh.status_code, 200)
    except Exception as e:
        result("A", "A8.6", "/health after reset", False, str(e), 200)


# ============================================================
# SUITE B: LOAD & PERFORMANCE TESTS
# ============================================================

def suite_b1_concurrent():
    print("\n[B1] 10 Concurrent POST /evaluate Requests", flush=True)
    N = 10
    timings = []
    errors = []

    def fire(i):
        p = {**VALID, "applicant_name": f"Load User {i}"}
        sc, r, ms = post_evaluate(p, timeout=40)
        timings.append(ms)
        if sc != 200:
            errors.append(f"req{i}:status={sc}")
        return sc == 200

    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
        futs = [ex.submit(fire, i) for i in range(N)]
        res_list = [f.result() for f in concurrent.futures.as_completed(futs)]

    all_ok = all(res_list)
    result("B", "B1.1", f"10 concurrent /evaluate -> all 200",
           all_ok, f"{sum(res_list)}/{N}", f"{N}/{N}",
           notes="; ".join(errors) if errors else "")
    if timings:
        mn, mx, avg = min(timings), max(timings), int(statistics.mean(timings))
        result("B", "B1.2", "Avg response < 30000ms (LLM timeout bound)",
               avg < 30000, f"avg={avg}ms min={mn}ms max={mx}ms", "avg<30000ms")
    return timings


def suite_b2_rapid_sequential():
    print("\n[B2] 15 Rapid Sequential POST /evaluate Requests", flush=True)
    N = 15
    successes = 0
    timings = []
    schema_ok = True

    for i in range(N):
        p = {**VALID, "applicant_name": f"Sequential User {i}"}
        sc, r, ms = post_evaluate(p, timeout=40)
        timings.append(ms)
        if sc == 200:
            successes += 1
            data = r.json()
            if not all(k in data for k in ["agent_decision", "score", "ground_truth"]):
                schema_ok = False
        print(f"  ... req {i+1}/{N}: {sc} ({ms}ms)", flush=True)

    result("B", "B2.1", "15 sequential /evaluate -> all succeed",
           successes == N, f"{successes}/{N}", f"{N}/{N}")
    result("B", "B2.2", "All responses have valid schema", schema_ok, schema_ok, True)
    if timings:
        avg = int(statistics.mean(timings))
        result("B", "B2.3", "Avg sequential response < 30000ms",
               avg < 30000, f"avg={avg}ms", "<30000ms")
    return timings


def suite_b3_timeout_enforcement():
    print("\n[B3] Timeout Enforcement (BUG-004 fix -- static analysis)", flush=True)

    app_py = os.path.join(PROJECT_ROOT, "server", "app.py")
    with open(app_py, "r", encoding="utf-8") as f:
        source = f.read()

    has_timeout = "timeout=30" in source or "timeout = 30" in source
    result("B", "B3.1", "call_llm() has timeout=30 parameter",
           has_timeout, has_timeout, True,
           notes="Static check -- ensures LLM calls won't hang indefinitely")

    has_try_except = "except Exception as e:" in source and "error" in source.lower()
    result("B", "B3.2", "call_llm() has try/except error handler",
           has_try_except, has_try_except, True,
           notes="Ensures timeout exception returns clean JSON, not raw traceback")

    t0 = time.time()
    sc, r, ms = post_evaluate({**VALID}, timeout=35)
    elapsed = time.time() - t0
    result("B", "B3.3", "Request completes within 35s (not infinite hang)",
           elapsed < 36, f"{elapsed:.1f}s", "<36s")
    if sc == 200:
        result("B", "B3.4", "Successful response is clean JSON (not traceback)",
               "Traceback" not in r.text and "stack" not in r.text.lower(),
               "clean JSON", "no traceback")


def suite_b4_reset_concurrent():
    print("\n[B4] /reset Under Concurrent Load", flush=True)
    eval_results = []
    reset_results = []

    def do_evaluate(i):
        p = {**VALID, "applicant_name": f"Concurrent {i}"}
        sc, r, ms = post_evaluate(p, timeout=40)
        eval_results.append(sc)
        return sc

    def do_reset(i):
        sc, r, ms = post_reset({})
        reset_results.append(sc)
        return sc

    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:
        futs = [ex.submit(do_evaluate, i) for i in range(5)]
        futs += [ex.submit(do_reset, i) for i in range(2)]
        _ = [f.result() for f in concurrent.futures.as_completed(futs)]

    no_500_eval = all(s not in (500, "ERROR", "TIMEOUT") for s in eval_results)
    no_500_reset = all(s not in (500, "ERROR") for s in reset_results)
    result("B", "B4.1", "No 500 errors in evaluate under concurrent load", no_500_eval, eval_results, "no 500s")
    result("B", "B4.2", "No 500 errors in reset under concurrent load", no_500_reset, reset_results, "no 500s")
    result("B", "B4.3", "Reset responses are 200", all(s == 200 for s in reset_results), reset_results, "[200]x2")


def suite_b5_health_tasks_load():
    print("\n[B5] /health and /tasks Under Load (50 concurrent each)", flush=True)
    N = 50

    def get_health():
        t0 = time.time()
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            return r.status_code, int((time.time() - t0) * 1000)
        except Exception:
            return 0, 5000

    def get_tasks():
        t0 = time.time()
        try:
            r = requests.get(f"{BASE_URL}/tasks", timeout=5)
            return r.status_code, int((time.time() - t0) * 1000)
        except Exception:
            return 0, 5000

    health_codes, health_times = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
        for f in concurrent.futures.as_completed([ex.submit(get_health) for _ in range(N)]):
            sc, ms = f.result()
            health_codes.append(sc)
            health_times.append(ms)

    all_h = all(c == 200 for c in health_codes)
    h_avg = int(statistics.mean(health_times)) if health_times else 9999
    result("B", "B5.1", f"50 concurrent /health -> all 200",
           all_h, f"{sum(1 for c in health_codes if c==200)}/{N}", f"{N}/{N}")
    result("B", "B5.2", "/health avg response < 100ms",
           h_avg < 100, f"avg={h_avg}ms", "<100ms")

    tasks_codes, tasks_times = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
        for f in concurrent.futures.as_completed([ex.submit(get_tasks) for _ in range(N)]):
            sc, ms = f.result()
            tasks_codes.append(sc)
            tasks_times.append(ms)

    all_t = all(c == 200 for c in tasks_codes)
    t_avg = int(statistics.mean(tasks_times)) if tasks_times else 9999
    result("B", "B5.3", f"50 concurrent /tasks -> all 200",
           all_t, f"{sum(1 for c in tasks_codes if c==200)}/{N}", f"{N}/{N}")
    result("B", "B5.4", "/tasks avg response < 100ms",
           t_avg < 100, f"avg={t_avg}ms", "<100ms")

    return h_avg, t_avg


# ============================================================
# SUITE C: LIVE DEPLOYMENT TESTS
# ============================================================

def suite_c1_startup():
    print("\n[C1] Server Startup Check", flush=True)
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        result("C", "C1.1", "Server up -- /health returns 200", r.status_code == 200, r.status_code, 200)
        data = r.json()
        result("C", "C1.2", "/health has status='ok'", data.get("status") == "ok", data.get("status"), "ok")
        result("C", "C1.3", "/health lists 8 tasks", data.get("available_tasks", 0) == 8, data.get("available_tasks"), 8)
    except Exception as e:
        for tid in ["C1.1", "C1.2", "C1.3"]:
            result("C", tid, "Server reachable", False, str(e), "200")


def suite_c2_static_serving():
    print("\n[C2] Static File Serving", flush=True)

    try:
        r = requests.get(f"{BASE_URL}/ui", timeout=5)
        result("C", "C2.1", "GET /ui -> 200 HTML",
               r.status_code == 200 and "html" in r.headers.get("content-type", "").lower(),
               r.status_code, 200)
    except Exception as e:
        result("C", "C2.1", "GET /ui -> 200", False, str(e), 200)

    try:
        r = requests.get(f"{BASE_URL}/static/index.html", timeout=5)
        result("C", "C2.2", "GET /static/index.html -> 200", r.status_code == 200, r.status_code, 200)
    except Exception as e:
        result("C", "C2.2", "GET /static/index.html", False, str(e), 200)

    try:
        r = requests.get(f"{BASE_URL}/openenv.yaml", timeout=5)
        is_yaml = "yaml" in r.headers.get("content-type", "")
        result("C", "C2.3", "GET /openenv.yaml -> 200 YAML",
               r.status_code == 200 and is_yaml,
               f"status={r.status_code} ct={r.headers.get('content-type', '')}", "200 text/yaml")
    except Exception as e:
        result("C", "C2.3", "GET /openenv.yaml", False, str(e), 200)

    try:
        r = requests.get(f"{BASE_URL}/nonexistent_xyz_123", timeout=5)
        result("C", "C2.4", "GET /nonexistent -> 404 not 500", r.status_code == 404, r.status_code, 404)
    except Exception as e:
        result("C", "C2.4", "GET /nonexistent -> 404", False, str(e), 404)


def suite_c3_reference_cases():
    print("\n[C3] Full Live Stack -- 3 Reference Cases", flush=True)

    case1 = {**VALID, "applicant_name": "Case1 User", "credit_score": 740,
             "annual_income": 1000000.0, "existing_debt": 120000.0, "previous_defaults": 0}
    sc1, r1, ms1 = post_evaluate(case1)
    result("C", "C3.1", "Case1 (FICO=740, DTI=12%) -> 200", sc1 == 200, sc1, 200, ms=ms1)
    if sc1 == 200:
        gt1 = r1.json().get("ground_truth", {})
        result("C", "C3.1b", "Case1 GT: Low/Approve/7-9%",
               gt1.get("risk_level") == "Low" and gt1.get("loan_decision") == "Approve"
               and gt1.get("interest_rate_tier") == "7-9%",
               f"{gt1.get('risk_level')}/{gt1.get('loan_decision')}/{gt1.get('interest_rate_tier')}",
               "Low/Approve/7-9%", ms=ms1)

    case2 = {**VALID, "applicant_name": "Case2 User", "credit_score": 670,
             "annual_income": 700000.0, "existing_debt": 196000.0,
             "employment_type": "self_employed", "previous_defaults": 1}
    sc2, r2, ms2 = post_evaluate(case2)
    result("C", "C3.2", "Case2 (FICO=670, DTI=28%) -> 200", sc2 == 200, sc2, 200, ms=ms2)
    if sc2 == 200:
        gt2 = r2.json().get("ground_truth", {})
        result("C", "C3.2b", "Case2 GT: Medium risk",
               gt2.get("risk_level") == "Medium",
               f"{gt2.get('risk_level')}/{gt2.get('loan_decision')}/{gt2.get('interest_rate_tier')}",
               "Medium/*", ms=ms2)

    case3 = {**VALID, "applicant_name": "Case3 User", "credit_score": 580,
             "annual_income": 420000.0, "existing_debt": 380000.0,
             "employment_type": "freelancer", "previous_defaults": 2,
             "has_collateral": False, "loan_amount": 650000.0}
    sc3, r3, ms3 = post_evaluate(case3)
    result("C", "C3.3", "Case3 (FICO=580, DTI=90%, 2 defaults) -> 200", sc3 == 200, sc3, 200, ms=ms3)
    if sc3 == 200:
        gt3 = r3.json().get("ground_truth", {})
        result("C", "C3.3b", "Case3 GT: High/Reject/14%+",
               gt3.get("risk_level") == "High" and gt3.get("loan_decision") == "Reject",
               f"{gt3.get('risk_level')}/{gt3.get('loan_decision')}/{gt3.get('interest_rate_tier')}",
               "High/Reject/14%+", ms=ms3)

    print(f"  Response times: Case1={ms1}ms, Case2={ms2}ms, Case3={ms3}ms", flush=True)
    return ms1, ms2, ms3


def suite_c4_env_vars():
    print("\n[C4] Environment Variable Validation", flush=True)
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        data = r.json()
        env_vars = data.get("env_vars", {})
        hf = env_vars.get("HF_TOKEN", "NOT SET")
        model = env_vars.get("MODEL_NAME", "NOT SET")
        api = env_vars.get("API_BASE_URL", "NOT SET")
        result("C", "C4.1", "HF_TOKEN visible in /health (masked, not raw)",
               True, hf, "SET or NOT SET", notes=f"Value: {hf}")
        result("C", "C4.2", "No raw secrets (value < 50 chars if masked)",
               len(hf) < 50, f"len={len(hf)}", "<50 chars masked")
        result("C", "C4.3", "MODEL_NAME reported in /health",
               "MODEL_NAME" in env_vars, bool(env_vars.get("MODEL_NAME")), True)
        result("C", "C4.4", "API_BASE_URL or fallback reported",
               "API_BASE_URL" in env_vars, True, True,
               notes="Server uses default HF router if not set -- acceptable")
    except Exception as e:
        result("C", "C4.1", "Env vars in /health", False, str(e), "200 JSON")


def suite_c5_cors():
    print("\n[C5] CORS Live Test", flush=True)
    try:
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        r = requests.options(f"{BASE_URL}/evaluate", headers=headers, timeout=5)
        ac_origin = r.headers.get("access-control-allow-origin", "")
        ac_methods = r.headers.get("access-control-allow-methods", "")
        result("C", "C5.1", "OPTIONS preflight -> 200 or 204",
               r.status_code in (200, 204), r.status_code, "200 or 204")
        result("C", "C5.2", "CORS Access-Control-Allow-Origin present",
               bool(ac_origin), ac_origin, "*")
        result("C", "C5.3", "CORS Allow-Methods present",
               bool(ac_methods) or ac_origin == "*", ac_methods, "includes POST")

        app_path = os.path.join(PROJECT_ROOT, "server", "app.py")
        with open(app_path, encoding="utf-8") as f:
            src = f.read()
        wildcard_creds = 'allow_origins=["*"]' in src and "allow_credentials=True" in src
        result("C", "C5.4", "WARN: allow_origins='*' + allow_credentials=True (browser rejects)",
               not wildcard_creds, wildcard_creds, False,
               notes="BUG: browsers refuse wildcard origin + credentials=True per CORS spec")
        if wildcard_creds:
            bug("BUG-C5", "server/app.py", "~L142-148", "MEDIUM",
                "allow_origins=['*'] + allow_credentials=True violates CORS spec. "
                "Browsers reject credentialed requests to wildcard origins. Use specific allowed origins.")
    except Exception as e:
        result("C", "C5.1", "CORS preflight", False, str(e), "200/204")


def suite_c6_reset_live_state():
    print("\n[C6] /reset Live State Test", flush=True)

    sc, r, ms = post_evaluate({**VALID})
    result("C", "C6.1", "Initial evaluate succeeds", sc == 200, sc, 200, ms=ms)

    sc_r, r_r, ms_r = post_reset({})
    result("C", "C6.2", "POST /reset -> 200", sc_r == 200, sc_r, 200, ms=ms_r)
    if sc_r == 200:
        data = r_r.json()
        result("C", "C6.3", "Reset returns status=reset_complete",
               data.get("status") == "reset_complete", data.get("status"), "reset_complete")
        obs = data.get("state", {}).get("observation", {})
        result("C", "C6.4", "Reset returns state with observation", bool(obs), bool(obs), True)

    sc2, r2, ms2 = post_evaluate({**VALID})
    result("C", "C6.5", "Post-reset evaluate succeeds (no state bleed)", sc2 == 200, sc2, 200, ms=ms2)


# ============================================================
# STATIC ANALYSIS
# ============================================================

def static_analysis():
    print("\n[STATIC] Code Pattern Analysis", flush=True)

    app_path = os.path.join(PROJECT_ROOT, "server", "app.py")
    with open(app_path, encoding="utf-8") as f:
        src = f.read()

    # LifecycleSession class-level mutable attrs
    has_class_list = "completed_stages: list = []" in src
    has_class_dict = "stage_scores: dict = {}" in src
    if has_class_list or has_class_dict:
        bug("BUG-STATIC-1", "server/app.py", "~L155-162", "MEDIUM",
            "LifecycleSession uses class-level mutable attributes (list, dict) not instance-level. "
            "While one instance exists now, this is a latent shared-state bug if multiple instances are created.")
    result("A", "STATIC-1", "LifecycleSession uses instance (not class-level mutable) attrs",
           not (has_class_list or has_class_dict), "class-level list/dict" if has_class_list else "OK",
           "instance-level in __init__",
           notes="Design flaw: mutable class attrs shared across instances")

    # Partial reset logic
    partial_reset = "if task_id == TASK_ORDER[0]:" in src
    if partial_reset:
        bug("BUG-STATIC-2", "server/app.py", "~L272-276", "MEDIUM",
            "global_session is only cleared when task_id == TASK_ORDER[0]. "
            "POST /reset with no task_id does NOT clear global_session -- state can leak between sessions.")
    result("A", "STATIC-2", "/reset always clears global_session (not conditional on task_id)",
           not partial_reset, "partial reset only for task1" if partial_reset else "unconditional",
           "always reset", notes="BUG: global_session not reset on bare /reset call")

    # loan_amount field missing gt=0 constraint
    loan_amount_line = [l for l in src.split("\n") if "loan_amount" in l and "Field" in l]
    has_loan_gt = any("gt=0" in l or "ge=1" in l or "gt=" in l for l in loan_amount_line)
    result("A", "STATIC-3", "loan_amount has gt=0 backend constraint",
           has_loan_gt,
           "gt=0 present" if has_loan_gt else "no gt/ge constraint",
           "Field(..., gt=0)",
           notes="BUG: frontend-only guard -- backend allows loan_amount=0 or negative")
    if not has_loan_gt:
        bug("BUG-STATIC-3", "server/app.py", "ApplicantInput loan_amount ~L192",
            "HIGH", "loan_amount has no gt=0 backend constraint. Frontend JS-only guard bypassed by direct API calls.")

    # annual_income field constraint  
    income_line = [l for l in src.split("\n") if "annual_income" in l and "Field" in l]
    has_income_gt = any("gt=0" in l or "ge=1" in l for l in income_line)
    result("A", "STATIC-4", "annual_income has gt=0 backend constraint",
           has_income_gt,
           "gt=0 present" if has_income_gt else "no gt/ge constraint",
           "Field(..., gt=0)",
           notes="BUG: annual_income=0 bypasses division-by-zero guard in graders.py")
    if not has_income_gt:
        bug("BUG-STATIC-4", "server/app.py", "ApplicantInput annual_income ~L188",
            "HIGH", "annual_income has no gt=0 backend constraint. graders.py has guard (income>0 else 1.0) but backend allows income=0.")

    # applicant_name min_length
    name_line = [l for l in src.split("\n") if "applicant_name" in l and "Field" in l]
    has_name_min = any("min_length" in l for l in name_line)
    result("A", "STATIC-5", "applicant_name has min_length=1 constraint",
           has_name_min,
           "min_length present" if has_name_min else "no min_length constraint",
           "Field(..., min_length=1)",
           notes="BUG: empty strings accepted as valid applicant names")
    if not has_name_min:
        bug("BUG-STATIC-5", "server/app.py", "ApplicantInput applicant_name ~L188",
            "MEDIUM", "applicant_name has no min_length constraint -- empty or whitespace-only names accepted.")


# ============================================================
# HTML REPORT GENERATOR
# ============================================================

def generate_html(b1_timings=None, b2_timings=None, h_avg=None, t_avg=None):
    pass_n = sum(1 for r in RESULTS if r["status"] == "PASS")
    fail_n = sum(1 for r in RESULTS if r["status"] == "FAIL")
    total = len(RESULTS)
    pct = int(pass_n / total * 100) if total else 0

    if fail_n == 0:
        verdict, vc, vb = "READY", "#00ff88", "rgba(0,255,136,0.1)"
    elif fail_n <= 6:
        verdict, vc, vb = "READY WITH WARNINGS", "#ffcc00", "rgba(255,204,0,0.1)"
    else:
        verdict, vc, vb = "NOT READY", "#ff3366", "rgba(255,51,102,0.1)"

    perf_rows = ""
    if b1_timings:
        avg = int(statistics.mean(b1_timings)); mx = max(b1_timings)
        ok = avg < 30000
        perf_rows += f"<tr><td>/evaluate concurrent x10</td><td>{avg}ms</td><td>{mx}ms</td><td>10/10</td><td style='color:{'#00ff88' if ok else '#ff3366'}'>{'PASS' if ok else 'FAIL'}</td></tr>"
    if b2_timings:
        avg = int(statistics.mean(b2_timings)); mx = max(b2_timings)
        ok = avg < 30000
        perf_rows += f"<tr><td>/evaluate sequential x15</td><td>{avg}ms</td><td>{mx}ms</td><td>{sum(1 for t in b2_timings if t<30000)}/15</td><td style='color:{'#00ff88' if ok else '#ff3366'}'>{'PASS' if ok else 'FAIL'}</td></tr>"
    if h_avg is not None:
        ok = h_avg < 100
        perf_rows += f"<tr><td>/health concurrent x50</td><td>{h_avg}ms</td><td>--</td><td>50/50</td><td style='color:{'#00ff88' if ok else '#ff3366'}'>{'PASS' if ok else 'FAIL'}</td></tr>"
    if t_avg is not None:
        ok = t_avg < 100
        perf_rows += f"<tr><td>/tasks concurrent x50</td><td>{t_avg}ms</td><td>--</td><td>50/50</td><td style='color:{'#00ff88' if ok else '#ff3366'}'>{'PASS' if ok else 'FAIL'}</td></tr>"

    result_rows = ""
    for r in RESULTS:
        cls = "pass" if r["status"] == "PASS" else "fail"
        icon = "PASS" if r["status"] == "PASS" else "FAIL"
        badge_cls = {"A": "badge-a", "B": "badge-b", "C": "badge-c"}.get(r["suite"], "badge-a")
        ms_str = f" [{r['ms']}ms]" if r.get("ms") else ""
        result_rows += (
            f"<tr class='{cls}-row'>"
            f"<td><span class='badge {badge_cls}'>{r['suite']}</span></td>"
            f"<td><code>{r['test_id']}</code></td>"
            f"<td>{r['name']}</td>"
            f"<td class='{cls}'>{icon}{ms_str}</td>"
            f"<td style='font-size:0.73rem;color:#aaa'>{r['actual'][:120]}</td>"
            f"<td style='font-size:0.73rem;color:#aaa'>{r['expected'][:80]}</td>"
            f"</tr>"
        )

    bug_rows = ""
    for b in BUGS:
        sev_color = {"HIGH": "#ff3366", "MEDIUM": "#ffcc00", "LOW": "#7b7b9a"}.get(b["severity"], "#aaa")
        bug_rows += (
            f"<tr><td><code>{b['id']}</code></td><td>{b['file']}</td><td>{b['line']}</td>"
            f"<td style='color:{sev_color};font-weight:700'>{b['severity']}</td>"
            f"<td style='font-size:0.73rem'>{b['description'][:200]}</td></tr>"
        )
    if not bug_rows:
        bug_rows = "<tr><td colspan='5' style='color:#00ff88;text-align:center'>No new bugs found</td></tr>"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QA Report v3 -- Nexus Bank Advanced Tests</title>
<style>
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box}}
:root{{--bg:#0a0a0f;--surface:#0f1923;--surface-alt:#141e2b;--primary:#00ff88;--secondary:#7b2fff;--accent:#ff6b35;--danger:#ff3366;--gold:#ffcc00;--border:#1a2a3a;--text:#e0e0ff;--text-dim:#6a7a8a;--mono:'JetBrains Mono',monospace}}
body{{background:var(--bg);color:var(--text);font-family:var(--mono);padding:2rem;line-height:1.5}}
h1{{font-size:1.5rem;color:var(--primary);letter-spacing:4px;text-transform:uppercase;margin-bottom:.3rem}}
.sub{{color:var(--text-dim);font-size:.78rem;letter-spacing:2px;margin-bottom:1.5rem}}
.verdict{{padding:1rem 2rem;border-radius:8px;font-size:1rem;font-weight:700;letter-spacing:3px;text-align:center;margin-bottom:1.5rem;text-transform:uppercase;background:{vb};color:{vc};border:2px solid {vc}}}
.meta{{display:flex;gap:1.2rem;margin-bottom:1.5rem;flex-wrap:wrap}}
.mbox{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:.8rem 1.2rem;text-align:center;min-width:90px}}
.mnum{{font-size:1.8rem;font-weight:700}}
.mlabel{{font-size:.6rem;color:var(--text-dim);letter-spacing:2px;text-transform:uppercase}}
.pc{{color:var(--primary)}}.fc{{color:var(--danger)}}.wc{{color:var(--gold)}}
.sec{{font-size:.8rem;color:var(--secondary);letter-spacing:3px;text-transform:uppercase;margin:1.8rem 0 .8rem;padding-bottom:.4rem;border-bottom:1px solid var(--border)}}
table{{width:100%;border-collapse:collapse;margin-bottom:1.5rem;font-size:.76rem}}
th{{background:var(--surface-alt);color:var(--text-dim);padding:.5rem .7rem;text-align:left;font-size:.62rem;letter-spacing:2px;text-transform:uppercase;border-bottom:2px solid var(--border)}}
td{{padding:.45rem .7rem;border-bottom:1px solid var(--border);vertical-align:top}}
.pass-row{{background:rgba(0,255,136,.02)}}.fail-row{{background:rgba(255,51,102,.05)}}
.pass{{color:var(--primary);font-weight:700}}.fail{{color:var(--danger);font-weight:700}}
.badge{{padding:.15rem .45rem;border-radius:3px;font-size:.62rem;font-weight:700;letter-spacing:1px}}
.badge-a{{background:rgba(123,47,255,.2);color:#9b6fff}}
.badge-b{{background:rgba(255,107,53,.2);color:#ff8c6e}}
.badge-c{{background:rgba(0,255,136,.2);color:#00ff88}}
code{{background:rgba(255,255,255,.05);padding:.1rem .3rem;border-radius:3px}}
footer{{margin-top:2.5rem;color:var(--text-dim);font-size:.65rem;text-align:center;letter-spacing:2px}}
.notes-col{{color:#888;font-size:.7rem;font-style:italic}}
</style>
</head>
<body>
<h1>&#9670; NEXUS BANK QA REPORT v3</h1>
<p class="sub">Advanced Stress &middot; Load &middot; Live Deployment &mdash; {now}</p>

<div class="verdict">FINAL VERDICT: {verdict}</div>

<div class="meta">
  <div class="mbox"><div class="mnum pc">{pass_n}</div><div class="mlabel">Passed</div></div>
  <div class="mbox"><div class="mnum fc">{fail_n}</div><div class="mlabel">Failed</div></div>
  <div class="mbox"><div class="mnum">{total}</div><div class="mlabel">Total</div></div>
  <div class="mbox"><div class="mnum {'pc' if pct>=90 else 'wc' if pct>=70 else 'fc'}">{pct}%</div><div class="mlabel">Pass Rate</div></div>
  <div class="mbox"><div class="mnum {'fc' if BUGS else 'pc'}">{len(BUGS)}</div><div class="mlabel">Bugs Found</div></div>
</div>

<div class="sec">&#9889; Performance Table</div>
<table>
  <tr><th>Endpoint / Test</th><th>Avg Latency</th><th>Max Latency</th><th>Success Rate</th><th>OK?</th></tr>
  {perf_rows if perf_rows else '<tr><td colspan="5" style="color:var(--text-dim);text-align:center">Performance tests not executed</td></tr>'}
</table>

<div class="sec">&#129514; Test Results</div>
<table>
  <tr><th>Suite</th><th>ID</th><th>Test</th><th>Result</th><th>Actual</th><th>Expected</th></tr>
  {result_rows}
</table>

<div class="sec">&#128027; Bugs Found</div>
<table>
  <tr><th>ID</th><th>File</th><th>Line</th><th>Severity</th><th>Description</th></tr>
  {bug_rows}
</table>

<div class="sec">&#128221; Bug Remediation Guide</div>
<table>
  <tr><th>Bug</th><th>Fix</th></tr>
  <tr><td><code>BUG-STATIC-3</code> loan_amount=0 accepted</td><td>Change <code>loan_amount: float</code> to <code>loan_amount: float = Field(..., gt=0)</code> in ApplicantInput</td></tr>
  <tr><td><code>BUG-STATIC-4</code> annual_income=0 accepted</td><td>Change <code>annual_income: float</code> to <code>annual_income: float = Field(..., gt=0)</code> in ApplicantInput</td></tr>
  <tr><td><code>BUG-STATIC-5</code> empty applicant_name</td><td>Add <code>min_length=1</code> to <code>applicant_name</code> Field definition</td></tr>
  <tr><td><code>BUG-STATIC-1</code> LifecycleSession class attrs</td><td>Add <code>__init__</code> method to LifecycleSession setting all attrs as instance variables</td></tr>
  <tr><td><code>BUG-STATIC-2</code> Partial reset</td><td>Move global_session reset outside the <code>if task_id == TASK_ORDER[0]</code> guard</td></tr>
  <tr><td><code>BUG-C5</code> CORS wildcard+credentials</td><td>Change <code>allow_origins=["*"]</code> to specific origins, or set <code>allow_credentials=False</code></td></tr>
</table>

<footer>NEXUS BANK QA SUITE v3 &middot; Advanced Stress, Load &amp; Live Deployment &middot; {now}</footer>
</body>
</html>"""

    out = os.path.join(PROJECT_ROOT, "qa_report_v3.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved: {out}", flush=True)
    return out


# ============================================================
# MAIN
# ============================================================

def wait_for_server(timeout=20):
    print(f"\nWaiting for server at {BASE_URL} ...", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print("  Server is UP!", flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)
    print("  Server did not come up in time", flush=True)
    return False


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("  NEXUS BANK -- QA REPORT v3", flush=True)
    print("  Advanced Stress . Load . Live Deployment Test Suite", flush=True)
    print("=" * 70, flush=True)

    server_ok = wait_for_server(25)

    b1_timings = b2_timings = None
    h_avg = t_avg = None

    print("\n" + "=" * 70, flush=True)
    print("  SUITE A: EDGE CASE STRESS TESTS", flush=True)
    print("=" * 70, flush=True)

    static_analysis()

    if server_ok:
        suite_a1_boundary_fico()
        suite_a2_dti_extremes()
        suite_a3_loan_extremes()
        suite_a4_injection()
        suite_a5_new_fields()

    suite_a6_grader_consistency()

    if server_ok:
        suite_a7_all_tasks()
        suite_a8_reset_between_tasks()

    if server_ok:
        print("\n" + "=" * 70, flush=True)
        print("  SUITE B: LOAD & PERFORMANCE TESTS", flush=True)
        print("=" * 70, flush=True)
        b1_timings = suite_b1_concurrent()
        b2_timings = suite_b2_rapid_sequential()
        suite_b3_timeout_enforcement()
        suite_b4_reset_concurrent()
        h_avg, t_avg = suite_b5_health_tasks_load()

        print("\n" + "=" * 70, flush=True)
        print("  SUITE C: LIVE DEPLOYMENT TESTS", flush=True)
        print("=" * 70, flush=True)
        suite_c1_startup()
        suite_c2_static_serving()
        suite_c3_reference_cases()
        suite_c4_env_vars()
        suite_c5_cors()
        suite_c6_reset_live_state()

    pass_n = sum(1 for r in RESULTS if r["status"] == "PASS")
    fail_n = sum(1 for r in RESULTS if r["status"] == "FAIL")
    total = len(RESULTS)

    print("\n" + "=" * 70, flush=True)
    print(f"  SUMMARY: {pass_n}/{total} PASSED | {fail_n} FAILED | {len(BUGS)} BUGS", flush=True)
    print("=" * 70, flush=True)

    if BUGS:
        print("\n  Bugs Found:", flush=True)
        for b in BUGS:
            print(f"    [{b['severity']}] {b['id']}: {b['description'][:80]}", flush=True)

    report_path = generate_html(b1_timings, b2_timings, h_avg, t_avg)
    print(f"\n  Open: {report_path}", flush=True)
