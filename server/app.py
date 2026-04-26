"""
OpenEnv-compliant server entry point for Loan Underwriting environment.

This module provides the FastAPI app and a `main()` function entry point
required by the [project.scripts] specification in pyproject.toml.

It wraps the existing app.py logic, providing all endpoints:
- GET  /              -> Health check (HTTP 200)
- GET  /health        -> Detailed health + env var status
- GET  /tasks         -> List all available tasks
- POST /reset         -> Reset environment, returns initial state
- POST /step          -> Submit action, returns (state, reward, done, info)
- GET  /state         -> Get current environment state
- GET  /openenv.yaml  -> Serve the OpenEnv spec file
- POST /grade         -> Grade a response for a given task_id
"""

import asyncio
from functools import partial
import os
import sys
import json
import logging
import re
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

from openai import OpenAI

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# ─── Path Setup ──────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so `environment` package can be found
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment import (
    LoanUnderwritingEnv,
    Action,
    RiskLevel,
    LoanDecision,
    InterestRateTier,
    ApplicantProfile,
    TASK_ORDER,
    grade_action,
    calculate_dynamic_ground_truth,
)
from environment.tasks import get_task

# ─── Logging Setup ───────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Environment Variable Validation ─────────────────────────────────────────

ENV_VARS = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]


def check_env_vars() -> dict:
    """Check which environment variables are configured."""
    status = {}
    for var in ENV_VARS:
        value = os.environ.get(var)
        if value:
            masked = value[:4] + "***" if len(value) > 4 else "***"
            status[var] = f"SET ({masked})"
        else:
            status[var] = "NOT SET"
    return status


# Log env var status at startup
env_status = check_env_vars()
for var, st in env_status.items():
    level = logging.INFO if "SET" in st and "NOT" not in st else logging.WARNING
    logger.log(level, f"  {var}: {st}")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set — LLM calls will fail with 401")

def _get_api_client():
    """
    Helper to initialize OpenAI client with correct Hugging Face pathing.
    Forces the use of the reliable global 'v1' endpoint.
    """
    base_url = os.environ.get("API_BASE_URL", "").strip()
    key = os.environ.get("HF_TOKEN", "").strip()
    model = os.getenv("MODEL_NAME", "Sourav0511/loan-underwriting-merged-v2").strip()

    # Use the reliable GLOBAL endpoint by default
    if not base_url or "api-inference.huggingface.co" in base_url:
        base_url = "https://router.huggingface.co/hf-inference/v1"

    logger.info(f"Final API Configuration: Model={model}, Endpoint={base_url}")
    if not key:
        logger.warning("⚠️ HF_TOKEN is empty. AI Agent will likely fail with 401 Unauthorized.")

    # Create client
    return OpenAI(base_url=base_url, api_key=HF_TOKEN if HF_TOKEN else "missing-token"), model


client, MODEL_NAME = _get_api_client()

async def call_llm(prompt: str, max_tokens: int = 300) -> dict:
    """Single LLM call with robust parsing for multi-stage chains."""
    try:
        local_client, local_model_name = _get_api_client()
        logger.info(f"Chain Stage: Calling {local_model_name}")
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(
                local_client.chat.completions.create,
                model=local_model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
                timeout=30
            )
        )
        raw = response.choices[0].message.content
        logger.info(f"Chain Stage: Received response ({len(raw)} chars)")
        return parse_llm_response(raw)
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Chain Stage Error: {err_msg}")
        if "Not Found" in err_msg:
            return {"error": err_msg, "reasoning": "Stage failed: Model not found. Check if MODEL_NAME secret is correct and public."}
        elif "Unauthorized" in err_msg or "401" in err_msg:
            return {"error": err_msg, "reasoning": "Stage failed: HF_TOKEN is missing or invalid. Check HF Secrets."}
        return {"error": err_msg, "reasoning": f"Stage failed: {err_msg}"}




evaluate_semaphore = asyncio.Semaphore(3)

# ─── FastAPI Application ─────────────────────────────────────────────────────

app = FastAPI(
    title="Loan Underwriting OpenEnv",
    description=(
        "An OpenEnv-compliant environment for Financial Loan Underwriting "
        "& Risk Assessment. Simulates a bank's loan underwriting desk where "
        "AI agents evaluate applicant profiles and make lending decisions."
    ),
    version="1.0.0",
)

# ─── CORS Middleware ─────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Environment Instance ─────────────────────────────────────────────

env = LoanUnderwritingEnv()
logger.info("LoanUnderwritingEnv initialized successfully.")

class LifecycleSession:
    def __init__(self):
        self.current_stage = 0
        self.completed_stages = []
        self.stage_scores = {}
        self.applicant_profile = {}
        self.total_score = 0.0

global_session = LifecycleSession()

# ─── Request/Response Models ─────────────────────────────────────────────────


class ResetRequest(BaseModel):
    """Request body for the /reset endpoint."""
    task_id: Optional[str] = None
    custom_profile: Optional[ApplicantProfile] = None


class StepRequest(BaseModel):
    """Request body for the /step endpoint."""
    risk_level: str
    loan_decision: str
    interest_rate_tier: str
    reasoning: Optional[str] = None


class GradeRequest(BaseModel):
    """Request body for the /grade endpoint."""
    task_id: str
    response: str


class ApplicantInput(BaseModel):
    """Request body for the /evaluate endpoint — applicant details only."""
    applicant_name: str = Field(..., min_length=1)
    annual_income: float = Field(..., gt=0, description="Annual income must be greater than 0")
    credit_score: int = Field(..., ge=300, le=850)
    existing_debt: float
    loan_amount: float = Field(..., gt=0, description="Loan amount must be greater than 0")
    employment_type: str
    employment_years: float = 0.0
    job_history_years: float = 0.0
    loan_tenure: int
    task_id: str
    age: int = 30
    monthly_expenses: float = 0.0
    has_collateral: bool = False
    previous_defaults: int = 0
    past_defaults: int = 0
    loan_purpose: str = "general"
    public_records: int = Field(0, ge=0)
    credit_inquiries_6mo: int = Field(0, ge=0)
    documents_submitted: Optional[List[str]] = None
    payment_history: Optional[List[str]] = None

    @field_validator("applicant_name")
    @classmethod
    def strip_name(cls, v):
        stripped = v.strip()
        if not stripped:
            raise ValueError("applicant_name cannot be empty or whitespace")
        return stripped


# ─── Health Check Endpoints ──────────────────────────────────────────────────


@app.get("/")
async def root():
    """Root endpoint. Returns HTTP 200 to confirm the service is running."""
    return {
        "status": "ok",
        "environment": "loan-underwriting-risk-assessment",
        "version": "1.0.0",
        "message": "Loan Underwriting OpenEnv is running.",
    }


@app.get("/health")
async def health_check():
    """Detailed health check with environment status and env var availability."""
    _, active_model = _get_api_client()
    return {
        "status": "ok",
        "environment": "loan-underwriting-risk-assessment",
        "version": "1.0.0",
        "model_name": active_model,
        "current_task": env.current_task_id,
        "is_done": env.is_done,
        "available_tasks": len(env.get_available_tasks()),
        "env_vars": check_env_vars(),
    }


# ─── Task Listing ────────────────────────────────────────────────────────────


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with their metadata."""
    return {
        "tasks": env.get_available_tasks(),
        "total": len(env.get_available_tasks()),
    }


# ─── Core OpenEnv Endpoints ──────────────────────────────────────────────────


@app.post("/reset")
async def reset_environment(request: Request):
    """
    Reset the environment and load a new applicant profile.

    Accepts:
    - POST with JSON body: {"task_id": "easy_salaried_high_credit"}
    - POST with empty body or no Content-Type (defaults to first task)
    """
    try:
        task_id = None
        custom_profile = None
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    task_id = body.get("task_id", None)
                    
                    global_session.completed_stages = []
                    global_session.stage_scores = {}
                    global_session.applicant_profile = {}
                    global_session.total_score = 0.0
                    global_session.current_stage = 0
                        
                    custom_profile_data = body.get("custom_profile", None)
                    if custom_profile_data:
                        custom_profile = ApplicantProfile(**custom_profile_data)
                    elif getattr(global_session, 'applicant_profile', None) and task_id != TASK_ORDER[0]:
                        try:
                            custom_profile = ApplicantProfile(**global_session.applicant_profile)
                        except Exception:
                            pass
            except Exception:
                pass
        state = env.reset(task_id=task_id, custom_profile=custom_profile)
        logger.info(f"Environment reset with task_id={task_id or 'default'} (custom={custom_profile is not None})")
        return {
            "status": "reset_complete",
            "state": state.model_dump(),
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step_environment(request: StepRequest):
    """
    Submit an underwriting decision and get grading results.
    Returns JSON with: state, reward (float 0.0-1.0), done (bool), info (dict).
    """
    try:
        action = Action(
            risk_level=request.risk_level,
            loan_decision=request.loan_decision,
            interest_rate_tier=request.interest_rate_tier,
            reasoning=request.reasoning,
        )
        state, reward, done, info = env.step(action)
        logger.info(
            f"Step completed: reward={reward:.3f}, "
            f"risk={request.risk_level}, decision={request.loan_decision}, "
            f"rate={request.interest_rate_tier}"
        )
        return {
            "state": state.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Step failed: {e}")
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
async def get_state():
    """Get the current environment state as JSON."""
    return {
        "state": env.state().model_dump(),
    }


# ─── Grade Endpoint (OpenEnv Spec) ───────────────────────────────────────────


def parse_response_to_action(response_text: str) -> Action:
    """
    Parse a free-text response string into a structured Action.

    Handles formats like:
    - "Low risk, Approve, 7-9%"
    - "High, Reject, 14%+"
    - JSON strings
    """
    text = response_text.strip()

    # Try JSON parsing first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return Action(
                risk_level=data.get("risk_level", "Medium"),
                loan_decision=data.get("loan_decision", "Conditional Approve"),
                interest_rate_tier=data.get("interest_rate_tier", "10-13%"),
                reasoning=data.get("reasoning"),
            )
    except (json.JSONDecodeError, ValueError):
        pass

    # Parse free-text: detect risk level
    risk_level = "Medium"
    text_lower = text.lower()
    if "low" in text_lower and "high" not in text_lower:
        risk_level = "Low"
    elif "high" in text_lower:
        risk_level = "High"
    elif "medium" in text_lower or "moderate" in text_lower:
        risk_level = "Medium"

    # Parse loan decision
    loan_decision = "Conditional Approve"
    if "reject" in text_lower:
        loan_decision = "Reject"
    elif "conditional" in text_lower:
        loan_decision = "Conditional Approve"
    elif "approve" in text_lower:
        loan_decision = "Approve"

    # Parse interest rate tier
    interest_rate_tier = "10-13%"
    if "7-9%" in text or "7-9" in text:
        interest_rate_tier = "7-9%"
    elif "14%+" in text or "14%" in text or "14+" in text:
        interest_rate_tier = "14%+"
    elif "10-13%" in text or "10-13" in text:
        interest_rate_tier = "10-13%"

    return Action(
        risk_level=risk_level,
        loan_decision=loan_decision,
        interest_rate_tier=interest_rate_tier,
        reasoning=f"Parsed from: {text[:200]}",
    )


@app.post("/grade")
async def grade_response(request: GradeRequest):
    """
    Grade a response for a given task.

    Accepts a task_id and a free-text response string.
    Parses the response, grades it against ground truth, and returns a score in [0.0, 1.0].
    """
    try:
        task_def = get_task(request.task_id)
    except KeyError:
        task_id_map = {
            "easy": "easy_salaried_high_credit",
            "medium": "medium_self_employed_moderate",
            "hard": "hard_freelancer_complex",
        }
        mapped_id = task_id_map.get(request.task_id.lower())
        if mapped_id:
            task_def = get_task(mapped_id)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task_id: '{request.task_id}'. "
                       f"Available: {TASK_ORDER} or shortcuts: {list(task_id_map.keys())}"
            )

    try:
        action = parse_response_to_action(request.response)
        grading_result = grade_action(action, task_def.ground_truth)
        logger.info(
            f"Grade request: task={request.task_id}, score={grading_result.total_score:.3f}"
        )
        return {
            "task_id": task_def.task_id,
            "score": grading_result.total_score,
            "feedback": grading_result.feedback,
            "grading": {
                "risk_level_score": grading_result.risk_level_score,
                "loan_decision_score": grading_result.loan_decision_score,
                "interest_rate_score": grading_result.interest_rate_score,
                "consistency_bonus": grading_result.consistency_bonus,
                "total_score": grading_result.total_score,
            },
            "parsed_action": {
                "risk_level": action.risk_level.value if hasattr(action.risk_level, 'value') else str(action.risk_level),
                "loan_decision": action.loan_decision.value if hasattr(action.loan_decision, 'value') else str(action.loan_decision),
                "interest_rate_tier": action.interest_rate_tier.value if hasattr(action.interest_rate_tier, 'value') else str(action.interest_rate_tier),
            },
        }
    except Exception as e:
        logger.error(f"Grade failed: {e}")
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")


# ─── Evaluate Endpoint (LLM-driven decision) ────────────────────────────────

import json
import re





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
    """
    Robustly parse LLM response even if formatting is off
    """
    # Try direct JSON parse first
    try:
        return json.loads(response_text)
    except:
        pass

    # Try extracting JSON from text
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
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

    return result

def stage_result_to_action(res: dict) -> Action:
    """Map intermediate LLM stage results to structured environment Actions."""
    # Attempt to extract fields if LLM returned them directly
    risk = res.get("risk_indicator") or res.get("risk_level")
    decision = res.get("preliminary_decision") or res.get("loan_decision")
    rate = res.get("interest_rate_tier") or res.get("processing_tier")
    
    # Fallbacks based on descriptive qualitative fields
    if not risk:
        status = res.get("document_status") or res.get("credit_character") or res.get("dti_assessment") or res.get("collateral_status")
        if status in ["Complete", "Strong", "Healthy", "Adequate"]: risk = "Low"
        elif status in ["Incomplete", "Average", "Borderline"]: risk = "Medium"
        else: risk = "High"
    
    if not decision:
        if risk == "Low": decision = "Approve"
        elif risk == "Medium": decision = "Conditional Approve"
        else: decision = "Reject"
        
    if not rate:
        if risk == "Low": rate = "7-9%"
        elif risk == "Medium": rate = "10-13%"
        else: rate = "14%+"
        
    return Action(
        risk_level=risk,
        loan_decision=decision,
        interest_rate_tier=rate,
        reasoning=res.get("reasoning", "Intermediate stage decision.")
    )

@app.post("/evaluate")
async def evaluate_applicant(applicant: ApplicantInput):
    # 1. Reset environment for a new full-lifecycle episode
    env.reset() # This starts the TASK_ORDER sequence
    
    # 2. Format profile for LLM context
    profile = f"""
    Applicant: {applicant.applicant_name} (Age: {getattr(applicant, 'age', 'N/A')})
    Employment: {applicant.employment_type} ({getattr(applicant, 'employment_years', 'N/A')} years)
    Income: ₹{applicant.annual_income:,.0f} | Debt: ₹{applicant.existing_debt:,.0f}
    Requested: ₹{applicant.loan_amount:,.0f} over {applicant.loan_tenure} months
    Credit Score: {applicant.credit_score} | Defaults: {getattr(applicant, 'previous_defaults', 0)}
    Collateral: {"Provided" if getattr(applicant, 'has_collateral', False) else "None"}
    """
    
    history = []
    stage_results = []
    total_env_reward = 0

    async with evaluate_semaphore:
        # ── STAGE 1: Documentation ──
        s1_prompt = f"Review identity and documents for: {profile}"
        s1_res = await call_llm(s1_prompt)
        # Convert LLM analysis to a real environment action
        action1 = stage_result_to_action(s1_res)
        _, r1, _, _ = env.step(action1)
        total_env_reward += r1
        stage_results.append({"stage": 1, "name": "Documentation", "result": s1_res, "reward": r1})
        history.append(f"Stage 1: {s1_res.get('reasoning', '')}")

        # ── STAGE 2: Credit ──
        s2_prompt = f"Analyze credit for: {profile}. Previous: {history[-1]}"
        s2_res = await call_llm(s2_prompt)
        action2 = stage_result_to_action(s2_res)
        _, r2, _, _ = env.step(action2)
        total_env_reward += r2
        stage_results.append({"stage": 2, "name": "Credit Character", "result": s2_res, "reward": r2})
        history.append(f"Stage 2: {s2_res.get('reasoning', '')}")

        # ── STAGE 3-4: Intermediate Processing ──
        # Advance env through intermediate lifecycle stages using accumulated context
        while env._current_step < 4:
            # For these background stages, we use the best available context from Stage 2
            env.step(action2)

        # ── STAGE 5: Final Verdict ──
        s5_prompt = f"Final verdict for: {profile}. History: {' | '.join(history)}"
        s5_res = await call_llm(s5_prompt, max_tokens=500)
        
        action_final = Action(
            risk_level=s5_res.get("risk_level", "Medium"),
            loan_decision=s5_res.get("loan_decision", "Conditional Approve"),
            interest_rate_tier=s5_res.get("interest_rate_tier", "10-13%"),
            reasoning=s5_res.get("reasoning", "")
        )
        
        state, r_final, done, info = env.step(action_final)
        total_env_reward += r_final
        stage_results.append({"stage": 5, "name": "Final Verdict", "result": s5_res, "reward": r_final})

    # Final Grading against dynamic ground truth
    p_grade = ApplicantProfile(
        applicant_name=applicant.applicant_name,
        annual_income=applicant.annual_income,
        credit_score=applicant.credit_score,
        existing_debt=applicant.existing_debt,
        loan_amount_requested=applicant.loan_amount,
        employment_type=applicant.employment_type,
        employment_years=applicant.employment_years,
        repayment_tenure_months=applicant.loan_tenure,
        age=applicant.age,
        monthly_expenses=applicant.monthly_expenses if applicant.monthly_expenses > 0 else 0.0,
        has_collateral=applicant.has_collateral,
        previous_defaults=applicant.previous_defaults
    )
    gt = calculate_dynamic_ground_truth(p_grade)
    
    return {
        "agent_decision": s5_res,
        "stage_results": stage_results,
        "score": total_env_reward / 3, # Average of the 3 main stages
        "ground_truth": {
            "risk_level": gt.risk_level.value,
            "loan_decision": gt.loan_decision.value,
            "interest_rate_tier": gt.interest_rate_tier.value,
            "explanation": gt.explanation
        },
        "status": "success",
        "grading": info.get("grading", {})
    }



# ─── OpenEnv Spec Endpoint ───────────────────────────────────────────────────


@app.get("/openenv.yaml")
async def get_openenv_spec():
    """Serve the OpenEnv specification YAML file."""
    yaml_path = os.path.join(PROJECT_ROOT, "openenv.yaml")
    if os.path.exists(yaml_path):
        return FileResponse(yaml_path, media_type="text/yaml")
    else:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")


# ─── Web UI Endpoint ────────────────────────────────────────────────────────

STATIC_DIR = os.path.join(PROJECT_ROOT, "static")


@app.get("/ui")
async def serve_ui():
    """Serve the Cyberpunk Enterprise Banking web UI."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(
        status_code=404,
        detail="UI not found. Ensure static/index.html exists."
    )


# Mount static files directory (after all routes to avoid intercepting API paths)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── Main Entry Point (used by [project.scripts]) ───────────────────────────


def main():
    """
    Start the FastAPI server using uvicorn.
    This is the entry point called by `server` in pyproject.toml.
    """
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting Loan Underwriting OpenEnv server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()