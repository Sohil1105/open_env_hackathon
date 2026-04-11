"""
Loan Underwriting OpenEnv Environment Package.

This package provides an OpenEnv-compliant reinforcement learning environment
for Financial Loan Underwriting & Risk Assessment.

Usage:
    from environment import LoanUnderwritingEnv, Action, RiskLevel, LoanDecision, InterestRateTier

    env = LoanUnderwritingEnv()
    state = env.reset("easy_salaried_high_credit")
    action = Action(
        risk_level=RiskLevel.LOW,
        loan_decision=LoanDecision.APPROVE,
        interest_rate_tier=InterestRateTier.LOW,
        reasoning="Strong profile with high credit score and stable employment."
    )
    state, reward, done, info = env.step(action)
"""

from .env import LoanUnderwritingEnv
from .models import (
    ApplicantProfile,
    Observation,
    Action,
    State,
    GroundTruth,
    GradingResult,
    RiskLevel,
    LoanDecision,
    InterestRateTier,
    EmploymentType,
    TaskDifficulty,
)
from .tasks import get_task, get_all_tasks, TASK_ORDER, ALL_TASKS, generate_heuristic_ground_truth
from .graders import grade_action
from .rewards import compute_reward, format_reward_breakdown

__all__ = [
    "LoanUnderwritingEnv",
    "ApplicantProfile",
    "Observation",
    "Action",
    "State",
    "GroundTruth",
    "GradingResult",
    "RiskLevel",
    "LoanDecision",
    "InterestRateTier",
    "EmploymentType",
    "TaskDifficulty",
    "get_task",
    "get_all_tasks",
    "TASK_ORDER",
    "ALL_TASKS",
    "grade_action",
    "compute_reward",
    "format_reward_breakdown",
    "generate_heuristic_ground_truth",
]
