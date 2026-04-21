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

import os
import sys
import json
import logging
import re
from typing import Optional, List

from openai import OpenAI

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

def _get_api_client():
    """
    Helper to initialize OpenAI client with correct Hugging Face pathing.
    Forces the use of the reliable global 'v1' endpoint.
    """
    base_url = os.environ.get("API_BASE_URL", "").strip()
    key = os.environ.get("HF_TOKEN", "").strip()
    model = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct").strip()

    # Use the reliable GLOBAL endpoint by default
    if not base_url or "api-inference.huggingface.co" in base_url:
        base_url = "https://api-inference.huggingface.co/v1"

    logger.info(f"Final API Configuration: Model={model}, Endpoint={base_url}")
    if not key:
        logger.warning("⚠️ HF_TOKEN is empty. AI Agent will likely fail with 401 Unauthorized.")

    # Create client
    return OpenAI(base_url=base_url, api_key=key if key else "missing-token"), model


client, MODEL_NAME = _get_api_client()



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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Environment Instance ─────────────────────────────────────────────

env = LoanUnderwritingEnv()
logger.info("LoanUnderwritingEnv initialized successfully.")

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
    applicant_name: str
    annual_income: float
    credit_score: int
    existing_debt: float
    loan_amount: float
    employment_type: str
    employment_years: float
    loan_tenure: int
    task_id: str
    age: int = 30
    monthly_expenses: float = 0.0
    has_collateral: bool = False
    previous_defaults: int = 0
    documents_submitted: Optional[List[str]] = None
    payment_history: Optional[List[str]] = None


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
    return {
        "status": "ok",
        "environment": "loan-underwriting-risk-assessment",
        "version": "1.0.0",
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
                    custom_profile_data = body.get("custom_profile", None)
                    if custom_profile_data:
                        custom_profile = ApplicantProfile(**custom_profile_data)
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


def _build_llm_prompt(applicant: "ApplicantInput") -> str:
    """Build the structured prompt sent to the LLM with applicant details."""
    dti = (applicant.existing_debt / applicant.annual_income * 100) if applicant.annual_income > 0 else 0
    lti = (applicant.loan_amount / applicant.annual_income * 100) if applicant.annual_income > 0 else 0

    return f"""You are an expert bank loan underwriter.
Analyze this applicant and make a decision.

APPLICANT DETAILS:
Name: {applicant.applicant_name}
Annual Income: ₹{applicant.annual_income:,.2f}
Credit Score: {applicant.credit_score}
Existing Debt: ₹{applicant.existing_debt:,.2f}
Loan Requested: ₹{applicant.loan_amount:,.2f}
Employment: {applicant.employment_type}
Loan Tenure: {applicant.loan_tenure} months
Debt-to-Income Ratio: {dti:.1f}%
Loan-to-Income Ratio: {lti:.1f}%

KEY RISK INDICATORS:
- Credit Score: {applicant.credit_score} {"(EXCELLENT 750+)" if applicant.credit_score >= 750 else "(GOOD 700-749)" if applicant.credit_score >= 700 else "(FAIR 620-699)" if applicant.credit_score >= 620 else "(POOR <620)"}
- DTI Ratio: {dti:.1f}% {"(LOW - Good)" if dti < 30 else "(MODERATE)" if dti < 50 else "(HIGH - Risky)"}
- Loan-to-Income: {lti:.1f}% {"(Manageable)" if lti < 100 else "(Stretched)" if lti < 200 else "(Overextended)"}

You MUST respond in this exact JSON format:
{{
  "risk_level": "Low" or "Medium" or "High",
  "loan_decision": "Approve" or "Conditional Approve" or "Reject",
  "interest_rate_tier": "7-9%" or "10-13%" or "14%+",
  "reasoning": "brief explanation of your decision"
}}

Only respond with the JSON. Nothing else."""


def _parse_llm_json(response_text: str) -> dict:
    """Extract and parse JSON from LLM response, handling markdown code blocks."""
    text = response_text.strip()
    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting the first JSON object from text
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Minimal fallback
    return {
        "risk_level": "Medium",
        "loan_decision": "Conditional Approve",
        "interest_rate_tier": "10-13%",
        "reasoning": f"Could not parse LLM response: {text[:200]}",
    }


@app.post("/evaluate")
async def evaluate_applicant(applicant: ApplicantInput):
    """
    End-to-end AI evaluation endpoint.

    Takes applicant details only, sends them to the LLM agent, parses the
    structured decision, grades it against a DYNAMIC ground truth computed
    from the actual details, and returns the AI's decision + score.
    """
    try:
        # 1. Create an ApplicantProfile from the input (handles dynamic stats)
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

        # 2. Reset the environment internally (to keep it consistent with UI stage)
        try:
            env.reset(task_id=applicant.task_id)
        except Exception:
            env.reset() # Fallback to default task

        # 3. Calculate DYNAMIC ground truth based on the PROVIDED details
        # This fixes the issue where changing credit score didn't change the expected answer.
        dynamic_gt = calculate_dynamic_ground_truth(profile)
        logger.info(f"Generated dynamic GT for {applicant.applicant_name}: risk={dynamic_gt.risk_level.value}")

        # 4. Build prompt and call LLM
        prompt = _build_llm_prompt(applicant)
        
        try:
            llm_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert bank loan underwriter. "
                            "Respond ONLY with the requested JSON object. No other text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            llm_text = llm_response.choices[0].message.content or ""
        except Exception as llm_err:
            # BROAD LOGGING for troubleshooting
            logger.error(f"LLM EXCEPTION: {type(llm_err).__name__}: {str(llm_err)}")
            
            reason = "LLM connection failed. Check your HF_TOKEN and API_BASE_URL."
            if "Authorization" in str(llm_err) or "401" in str(llm_err):
                reason = "LLM Unauthorized: HF_TOKEN is likely missing or invalid."
            elif "404" in str(llm_err):
                reason = f"LLM Model Not Found at {client.base_url}. Model {MODEL_NAME} may be unavailable."
            
            llm_text = json.dumps({
                "risk_level": "Medium",
                "loan_decision": "Conditional Approve",
                "interest_rate_tier": "10-13%",
                "reasoning": f"⚠️ {reason}"
            })

        # 5. Parse LLM response into structured dict
        parsed = _parse_llm_json(llm_text)

        # 6. Build Action and grade using the DYNAMIC ground truth
        action = Action(
            risk_level=parsed.get("risk_level", "Medium"),
            loan_decision=parsed.get("loan_decision", "Conditional Approve"),
            interest_rate_tier=parsed.get("interest_rate_tier", "10-13%"),
            reasoning=parsed.get("reasoning", ""),
        )
        
        # Use grade_action directly with our dynamic_gt
        grading_result = grade_action(action, dynamic_gt)

        # 7. Return AI decision + score + dynamic GT info
        return {
            "agent_decision": {
                "risk_level": action.risk_level.value,
                "loan_decision": action.loan_decision.value,
                "interest_rate_tier": action.interest_rate_tier.value,
                "reasoning": parsed.get("reasoning", ""),
            },
            "score": grading_result.total_score,
            "correct_answer": {
                "risk_level": dynamic_gt.risk_level.value,
                "loan_decision": dynamic_gt.loan_decision.value,
                "interest_rate_tier": dynamic_gt.interest_rate_tier.value,
                "explanation": dynamic_gt.explanation
            },
            "stage": applicant.task_id,
            "grading": {
                "risk_level_score": grading_result.risk_level_score,
                "loan_decision_score": grading_result.loan_decision_score,
                "interest_rate_score": grading_result.interest_rate_score,
                "consistency_bonus": grading_result.consistency_bonus,
            },
            "feedback": grading_result.feedback,
            "task_name": "Dynamic Case Assessment",
            "task_difficulty": "adaptive",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/evaluate failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


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