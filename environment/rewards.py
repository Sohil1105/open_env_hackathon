"""
Reward computation for the Loan Underwriting OpenEnv environment.

Reward is non-binary: weighted sum of per-component accuracy scores plus a
consistency bonus/penalty, clamped to (0.01, 0.99).
"""

from .models import Action, GroundTruth, GradingResult, RiskLevel, LoanDecision, InterestRateTier
from .graders import grade_action


# Weights match openenv.yaml
RISK_WEIGHT = 0.40
DECISION_WEIGHT = 0.35
RATE_WEIGHT = 0.25


def compute_reward(action: Action, ground_truth: GroundTruth) -> tuple[float, GradingResult]:
    """
    Compute the reward for an agent's action given the ground truth.
    """
    grading_result = grade_action(action, ground_truth)

    base_score = (
        grading_result.risk_level_score * RISK_WEIGHT +
        grading_result.loan_decision_score * DECISION_WEIGHT +
        grading_result.interest_rate_score * RATE_WEIGHT
    )

    consistency_bonus = grading_result.consistency_bonus
    reward = max(0.01, min(0.99, base_score + consistency_bonus))

    grading_result.consistency_bonus = consistency_bonus
    grading_result.total_score = reward

    return reward, grading_result


def compute_component_rewards(action: Action, ground_truth: GroundTruth) -> dict[str, float]:
    """Return per-component scores for diagnostic analysis across episodes."""
    _, grading = compute_reward(action, ground_truth)

    return {
        "risk_level": grading.risk_level_score,
        "loan_decision": grading.loan_decision_score,
        "interest_rate": grading.interest_rate_score,
        "consistency": grading.consistency_bonus,
        "total": grading.total_score,
        "risk_contribution": grading.risk_level_score * RISK_WEIGHT,
        "decision_contribution": grading.loan_decision_score * DECISION_WEIGHT,
        "rate_contribution": grading.interest_rate_score * RATE_WEIGHT,
    }


def format_reward_breakdown(grading_result: GradingResult) -> str:
    """Format a human-readable reward breakdown for logging."""
    lines = [
        "┌─────────────────────────────────────────────┐",
        "│         REWARD BREAKDOWN                     │",
        "├─────────────────────────────────────────────┤",
        f"│ Risk Level:     {grading_result.risk_level_score:.2f} × 0.40 = "
        f"{grading_result.risk_level_score * RISK_WEIGHT:.3f}  │",
        f"│ Loan Decision:  {grading_result.loan_decision_score:.2f} × 0.35 = "
        f"{grading_result.loan_decision_score * DECISION_WEIGHT:.3f}  │",
        f"│ Interest Rate:  {grading_result.interest_rate_score:.2f} × 0.25 = "
        f"{grading_result.interest_rate_score * RATE_WEIGHT:.3f}  │",
        f"│ Consistency:    {grading_result.consistency_bonus:+.2f}             │",
        "├─────────────────────────────────────────────┤",
        f"│ TOTAL REWARD:   {grading_result.total_score:.3f}                    │",
        "└─────────────────────────────────────────────┘",
    ]
    return "\n".join(lines)
