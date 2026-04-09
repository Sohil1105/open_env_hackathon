"""
FastAPI server exposing the Loan Underwriting OpenEnv as a REST API.

Deployed on Hugging Face Spaces (Docker SDK, port 7860).

Endpoints:
- GET  /              -> Health check (returns HTTP 200)
- GET  /health        -> Detailed health + env var status
- GET  /tasks         -> List all available tasks
- POST /reset         -> Reset environment with a task_id, returns initial state
- POST /step          -> Submit an action, returns (state, reward, done, info)
- GET  /state         -> Get current environment state
- GET  /openenv.yaml  -> Serve the OpenEnv spec file
- POST /grade       -> Grade a response for a given task_id
"""

import os
import re
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from environment import (
    LoanUnderwritingEnv,
    Action,
    RiskLevel,
    LoanDecision,
    InterestRateTier,
    ApplicantProfile,
    TASK_ORDER,
    grade_action,
)
from environment.tasks import get_task

# ─── Logging Setup ───────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Environment Variable Validation ─────────────────────────────────────────
# These are read by inference.py at runtime, not by the server itself.
# We log warnings at startup so operators know if secrets are missing.

ENV_VARS = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]

def check_env_vars() -> dict:
    """Check which environment variables are configured."""
    status = {}
    for var in ENV_VARS:
        value = os.environ.get(var)
        if value:
            # Mask the value for security (show first 4 chars only)
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

# ─── CORS Middleware — allow all origins for HF Spaces compatibility ─────────

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


# ─── Health Check Endpoints ──────────────────────────────────────────────────

@app.get("/")
async def root():
    """
    Root endpoint. Returns HTTP 200 to confirm the service is running.
    This is the primary endpoint checked by HF Spaces health monitoring.
    """
    return {
        "status": "ok",
        "environment": "loan-underwriting-risk-assessment",
        "version": "1.0.0",
        "message": "Loan Underwriting OpenEnv is running.",
    }


@app.get("/health")
async def health_check():
    """
    Detailed health check with environment status and env var availability.
    Returns {"status": "ok"} plus diagnostic information.
    """
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

    If task_id is not provided, defaults to the first (easy) task.
    Returns the initial State as JSON.
    """
    try:
        # Handle both JSON body and empty/no body gracefully
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

    Requires the environment to be reset first (call POST /reset before POST /step).

    Returns JSON with: state, reward (float 0.0-1.0), done (bool), info (dict).
    """
    try:
        # Build the Action from the request
        action = Action(
            risk_level=request.risk_level,
            loan_decision=request.loan_decision,
            interest_rate_tier=request.interest_rate_tier,
            reasoning=request.reasoning,
        )

        # Step the environment
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

class GradeRequest(BaseModel):
    """Request body for the /grade endpoint."""
    task_id: str
    response: str


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
        import json
        data = json.loads(text)
        if isinstance(data, dict):
            return Action(
                risk_level=data.get("risk_level", "Medium"),
                loan_decision=data.get("loan_decision", "Conditional Approve"),
                interest_rate_tier=data.get("interest_rate_tier", "10-13%"),
                reasoning=data.get("reasoning"),
            )
    except (Exception,):
        pass

    # Parse free-text: detect risk level
    risk_level = "Medium"  # default
    text_lower = text.lower()
    if "low" in text_lower and "high" not in text_lower:
        risk_level = "Low"
    elif "high" in text_lower:
        risk_level = "High"
    elif "medium" in text_lower or "moderate" in text_lower:
        risk_level = "Medium"

    # Parse loan decision
    loan_decision = "Conditional Approve"  # default
    if "reject" in text_lower:
        loan_decision = "Reject"
    elif "conditional" in text_lower:
        loan_decision = "Conditional Approve"
    elif "approve" in text_lower:
        loan_decision = "Approve"

    # Parse interest rate tier
    interest_rate_tier = "10-13%"  # default
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

    This endpoint is required by the OpenEnv spec for automated evaluation.
    """
    try:
        # Validate task_id exists
        task_def = get_task(request.task_id)
    except KeyError as e:
        # Try mapping short names to full task IDs
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
        # Parse the response into an Action
        action = parse_response_to_action(request.response)

        # Grade the action against ground truth
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


# ─── OpenEnv Spec Endpoint ───────────────────────────────────────────────────

@app.get("/openenv.yaml")
async def get_openenv_spec():
    """Serve the OpenEnv specification YAML file."""
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openenv.yaml")
    if os.path.exists(yaml_path):
        return FileResponse(yaml_path, media_type="text/yaml")
    else:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")


# ─── Main Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
