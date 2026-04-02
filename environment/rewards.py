"""
Reward function logic for the Loan Underwriting OpenEnv environment.

Computes partial progress reward signals based on the grading results.
The reward is NOT binary — it reflects how many sub-components the agent
answered correctly and how logically consistent the full decision is.
"""

from .models import Action, GroundTruth, GradingResult
from .graders import grade_action


# ─── Reward Weights ──────────────────────────────────────────────────────────

# These weights determine how much each component contributes to the reward.
# They match the scoring weights defined in openenv.yaml.
RISK_WEIGHT = 0.40
DECISION_WEIGHT = 0.35
RATE_WEIGHT = 0.25


def compute_reward(action: Action, ground_truth: GroundTruth) -> tuple[float, GradingResult]:
    """
    Compute the reward for an agent's action given the ground truth.

    The reward function provides partial progress signals:
    - Each correct sub-component adds its weighted portion to the reward
    - Partial credit is awarded for "close" answers (adjacent categories)
    - Logical consistency between the three decisions adds a small bonus/penalty
    - The final reward is guaranteed to be in [0.0, 1.0]

    Args:
        action: The agent's underwriting decision
        ground_truth: The correct answers for this applicant

    Returns:
        Tuple of (reward_float, grading_result) where reward is in [0.0, 1.0]
    """
    # Use the grading system to get detailed scoring
    grading_result = grade_action(action, ground_truth)

    # The total_score from the grader IS the reward — it already includes
    # weighted components and consistency bonuses, clamped to [0.0, 1.0]
    reward = grading_result.total_score

    return reward, grading_result


def compute_component_rewards(action: Action, ground_truth: GroundTruth) -> dict[str, float]:
    """
    Compute individual reward signals for each decision component.

    This is useful for detailed analysis of which components the agent
    is getting right or wrong across multiple episodes.

    Returns:
        Dictionary mapping component names to their individual scores (0.0–1.0)
    """
    _, grading = compute_reward(action, ground_truth)

    return {
        "risk_level": grading.risk_level_score,
        "loan_decision": grading.loan_decision_score,
        "interest_rate": grading.interest_rate_score,
        "consistency": grading.consistency_bonus,
        "total": grading.total_score,
        # Weighted contributions to final score
        "risk_contribution": grading.risk_level_score * RISK_WEIGHT,
        "decision_contribution": grading.loan_decision_score * DECISION_WEIGHT,
        "rate_contribution": grading.interest_rate_score * RATE_WEIGHT,
    }


def format_reward_breakdown(grading_result: GradingResult) -> str:
    """
    Format a human-readable reward breakdown string.

    Useful for logging and debugging agent performance.
    """
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
