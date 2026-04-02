"""
Automated graders for the Loan Underwriting OpenEnv environment.

Each grader evaluates one component of the agent's decision and returns
a float score in [0.0, 1.0] with partial credit:

- Risk Level Grader: 0.4 weight — exact match or partial for adjacent level
- Loan Decision Grader: 0.35 weight — exact match or partial for adjacent decision
- Interest Rate Grader: 0.25 weight — exact match or partial for adjacent tier
- Consistency Grader: bonus/penalty for logical alignment across all three decisions

Edge-case safe: all graders handle None, empty strings, and invalid values
by returning 0.0 instead of crashing.
"""

from .models import (
    Action,
    GroundTruth,
    GradingResult,
    RiskLevel,
    LoanDecision,
    InterestRateTier,
)


# ─── Ordinal mappings for computing distance between categories ──────────────

RISK_LEVEL_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
}

LOAN_DECISION_ORDER = {
    LoanDecision.APPROVE: 0,
    LoanDecision.CONDITIONAL_APPROVE: 1,
    LoanDecision.REJECT: 2,
}

INTEREST_RATE_ORDER = {
    InterestRateTier.LOW: 0,
    InterestRateTier.MEDIUM: 1,
    InterestRateTier.HIGH: 2,
}


# ─── Component Graders ───────────────────────────────────────────────────────

def grade_risk_level(predicted: RiskLevel, expected: RiskLevel) -> float:
    """
    Grade the risk level classification.

    Scoring:
    - Exact match: 1.0
    - Off by one level (e.g., Low vs Medium): 0.3
    - Off by two levels (e.g., Low vs High): 0.0
    - Invalid/None input: 0.0

    Returns: float in [0.0, 1.0]
    """
    # Edge-case: handle None or invalid enum values
    if predicted is None or expected is None:
        return 0.0
    if predicted not in RISK_LEVEL_ORDER or expected not in RISK_LEVEL_ORDER:
        return 0.0

    pred_ord = RISK_LEVEL_ORDER[predicted]
    exp_ord = RISK_LEVEL_ORDER[expected]
    distance = abs(pred_ord - exp_ord)

    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.3  # Partial credit for adjacent classification
    else:
        return 0.0  # Completely wrong


def grade_loan_decision(predicted: LoanDecision, expected: LoanDecision) -> float:
    """
    Grade the loan approval decision.

    Scoring:
    - Exact match: 1.0
    - Off by one step (e.g., Approve vs Conditional): 0.35
    - Off by two steps (e.g., Approve vs Reject): 0.0
    - Invalid/None input: 0.0

    Returns: float in [0.0, 1.0]
    """
    # Edge-case: handle None or invalid enum values
    if predicted is None or expected is None:
        return 0.0
    if predicted not in LOAN_DECISION_ORDER or expected not in LOAN_DECISION_ORDER:
        return 0.0

    pred_ord = LOAN_DECISION_ORDER[predicted]
    exp_ord = LOAN_DECISION_ORDER[expected]
    distance = abs(pred_ord - exp_ord)

    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.35  # Partial credit — at least in the right direction
    else:
        return 0.0  # Completely wrong (approve when should reject, or vice versa)


def grade_interest_rate(predicted: InterestRateTier, expected: InterestRateTier) -> float:
    """
    Grade the interest rate tier recommendation.

    Scoring:
    - Exact match: 1.0
    - Off by one tier (e.g., 7-9% vs 10-13%): 0.3
    - Off by two tiers (e.g., 7-9% vs 14%+): 0.0
    - Invalid/None input: 0.0

    Returns: float in [0.0, 1.0]
    """
    # Edge-case: handle None or invalid enum values
    if predicted is None or expected is None:
        return 0.0
    if predicted not in INTEREST_RATE_ORDER or expected not in INTEREST_RATE_ORDER:
        return 0.0

    pred_ord = INTEREST_RATE_ORDER[predicted]
    exp_ord = INTEREST_RATE_ORDER[expected]
    distance = abs(pred_ord - exp_ord)

    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.3  # Partial credit for adjacent tier
    else:
        return 0.0  # Completely wrong


def grade_consistency(action: Action) -> float:
    """
    Grade the logical consistency across all three decisions.

    Rules for consistency:
    - Low risk should pair with Approve and low interest rate
    - Medium risk should pair with Conditional Approve and medium interest rate
    - High risk should pair with Reject or Conditional Approve and high interest rate
    - Contradictions (e.g., Low risk + Reject, or High risk + Approve at 7-9%)
      incur a penalty

    Returns: float in [-0.1, 0.1] as a bonus/penalty modifier
    """
    # Edge-case: handle None fields
    if action is None:
        return 0.0

    risk = action.risk_level
    decision = action.loan_decision
    rate = action.interest_rate_tier

    if risk is None or decision is None or rate is None:
        return 0.0

    # Define what's logically consistent for each risk level
    consistency_score = 0.0

    if risk == RiskLevel.LOW:
        # Low risk: Approve at low rate is most consistent
        if decision == LoanDecision.APPROVE:
            consistency_score += 0.05
        elif decision == LoanDecision.REJECT:
            consistency_score -= 0.05  # Contradictory

        if rate == InterestRateTier.LOW:
            consistency_score += 0.05
        elif rate == InterestRateTier.HIGH:
            consistency_score -= 0.05  # Contradictory

    elif risk == RiskLevel.MEDIUM:
        # Medium risk: Conditional approve at medium rate is most consistent
        if decision == LoanDecision.CONDITIONAL_APPROVE:
            consistency_score += 0.05
        elif decision == LoanDecision.APPROVE and rate == InterestRateTier.LOW:
            consistency_score -= 0.05  # Too generous for medium risk

        if rate == InterestRateTier.MEDIUM:
            consistency_score += 0.05

    elif risk == RiskLevel.HIGH:
        # High risk: Reject or Conditional at high rate is most consistent
        if decision in (LoanDecision.REJECT, LoanDecision.CONDITIONAL_APPROVE):
            consistency_score += 0.05
        elif decision == LoanDecision.APPROVE:
            consistency_score -= 0.05  # Approving a high-risk applicant

        if rate == InterestRateTier.HIGH:
            consistency_score += 0.05
        elif rate == InterestRateTier.LOW:
            consistency_score -= 0.05  # Low rate for high risk is contradictory

    # Clamp to [-0.1, 0.1]
    return max(-0.1, min(0.1, consistency_score))


# ─── Main Grading Function ───────────────────────────────────────────────────

def grade_action(action: Action, ground_truth: GroundTruth) -> GradingResult:
    """
    Grade a complete agent action against the ground truth.

    Weights:
    - Risk level:     0.40 (40% of total score)
    - Loan decision:  0.35 (35% of total score)
    - Interest rate:  0.25 (25% of total score)
    - Consistency:    bonus/penalty modifier

    Handles edge cases: if action or ground_truth is None, returns score 0.0.

    Returns: GradingResult with detailed breakdown and total score in [0.0, 1.0]
    """
    # Edge-case: if action or ground_truth is None, return zero score
    if action is None or ground_truth is None:
        return GradingResult(
            risk_level_score=0.0,
            loan_decision_score=0.0,
            interest_rate_score=0.0,
            consistency_bonus=0.0,
            total_score=0.0,
            feedback="❌ No valid action or ground truth provided.",
        )

    # Grade each component
    risk_score = grade_risk_level(action.risk_level, ground_truth.risk_level)
    decision_score = grade_loan_decision(action.loan_decision, ground_truth.loan_decision)
    rate_score = grade_interest_rate(action.interest_rate_tier, ground_truth.interest_rate_tier)
    consistency = grade_consistency(action)

    # Calculate weighted total
    weighted_total = (
        risk_score * 0.40 +
        decision_score * 0.35 +
        rate_score * 0.25 +
        consistency
    )

    # Clamp final score to [0.0, 1.0] — guarantees output is always valid
    total_score = max(0.0, min(1.0, weighted_total))

    # Generate human-readable feedback
    feedback_parts = []

    # Risk level feedback
    if risk_score == 1.0:
        feedback_parts.append(f"✅ Risk level: Correct ({action.risk_level.value})")
    elif risk_score > 0:
        feedback_parts.append(
            f"⚠️ Risk level: Partially correct — predicted {action.risk_level.value}, "
            f"expected {ground_truth.risk_level.value}"
        )
    else:
        feedback_parts.append(
            f"❌ Risk level: Incorrect — predicted {action.risk_level.value}, "
            f"expected {ground_truth.risk_level.value}"
        )

    # Loan decision feedback
    if decision_score == 1.0:
        feedback_parts.append(f"✅ Loan decision: Correct ({action.loan_decision.value})")
    elif decision_score > 0:
        feedback_parts.append(
            f"⚠️ Loan decision: Partially correct — predicted {action.loan_decision.value}, "
            f"expected {ground_truth.loan_decision.value}"
        )
    else:
        feedback_parts.append(
            f"❌ Loan decision: Incorrect — predicted {action.loan_decision.value}, "
            f"expected {ground_truth.loan_decision.value}"
        )

    # Interest rate feedback
    if rate_score == 1.0:
        feedback_parts.append(
            f"✅ Interest rate: Correct ({action.interest_rate_tier.value})"
        )
    elif rate_score > 0:
        feedback_parts.append(
            f"⚠️ Interest rate: Partially correct — predicted {action.interest_rate_tier.value}, "
            f"expected {ground_truth.interest_rate_tier.value}"
        )
    else:
        feedback_parts.append(
            f"❌ Interest rate: Incorrect — predicted {action.interest_rate_tier.value}, "
            f"expected {ground_truth.interest_rate_tier.value}"
        )

    # Consistency feedback
    if consistency > 0:
        feedback_parts.append(f"🔗 Consistency bonus: +{consistency:.2f}")
    elif consistency < 0:
        feedback_parts.append(f"⚠️ Consistency penalty: {consistency:.2f} (decisions are logically contradictory)")

    feedback = "\n".join(feedback_parts)

    return GradingResult(
        risk_level_score=risk_score,
        loan_decision_score=decision_score,
        interest_rate_score=rate_score,
        consistency_bonus=consistency,
        total_score=total_score,
        feedback=feedback,
    )
