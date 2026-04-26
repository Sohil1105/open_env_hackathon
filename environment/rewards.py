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
    """Format a professional, high-visibility audit log for the terminal."""
    consistency = grading_result.consistency_bonus
    
    # Visual cues for the terminal
    status_icon = "🟢" if grading_result.total_score > 0.8 else "🟡" if grading_result.total_score > 0.5 else "🔴"
    integrity_status = "STABLE" if consistency >= 0 else "COMPROMISED" if consistency <= -0.15 else "WARNING"
    
    lines = [
        "\n" + "="*60,
        f" FINANCIAL AUDIT LOG | SCORE: {grading_result.total_score:.3f} {status_icon}",
        "="*60,
        f"  [COMPONENTS]",
        f"  • Risk Assessment:    {grading_result.risk_level_score:.2f} (Weight: 40%)",
        f"  • Loan Decision:      {grading_result.loan_decision_score:.2f} (Weight: 35%)",
        f"  • Interest Rate:      {grading_result.interest_rate_score:.2f} (Weight: 25%)",
        "",
        f"  [INTEGRITY CHECK: {integrity_status}]",
        f"  • Consistency Bonus:  {consistency:+.2f}",
    ]
    
    if consistency <= -0.15:
        lines.append("  ! CRITICAL: Irrational Pricing detected (High Risk + Low Rate)")
    elif consistency < 0:
        lines.append("  ! WARNING: Minor logical contradiction in underwriting")
    elif consistency > 0.10:
        lines.append("  * EXCELLENT: Perfect risk-adjusted alignment")
        
    lines.append("="*60 + "\n")
    return "\n".join(lines)
