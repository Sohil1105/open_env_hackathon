"""
Graders for the Loan Underwriting OpenEnv environment.

Each grader returns a float in [0.0, 1.0] using ordinal distance scoring:
exact match → 1.0, off-by-one → partial credit, off-by-two → 0.01.
All graders are None-safe and clamp output to [0.01, 0.99].
"""

from .models import (
    Action,
    GroundTruth,
    GradingResult,
    RiskLevel,
    LoanDecision,
    InterestRateTier,
)

RISK_SIMILARITY = {
    ("low", "low"): 0.99,
    ("low", "medium"): 0.3,
    ("low", "high"): 0.01,
    ("medium", "medium"): 0.99,
    ("medium", "low"): 0.3,
    ("medium", "high"): 0.3,
    ("high", "high"): 0.99,
    ("high", "medium"): 0.3,
    ("high", "low"): 0.01,
}

DECISION_SIMILARITY = {
    ("approve", "approve"): 0.99,
    ("approve", "conditional approve"): 0.4,
    ("approve", "reject"): 0.01,
    ("conditional approve", "conditional approve"): 0.99,
    ("conditional approve", "approve"): 0.4,
    ("conditional approve", "reject"): 0.4,
    ("reject", "reject"): 0.99,
    ("reject", "conditional approve"): 0.4,
    ("reject", "approve"): 0.01,
}

RATE_SIMILARITY = {
    ("7-9%", "7-9%"): 0.99,
    ("7-9%", "10-13%"): 0.3,
    ("7-9%", "14%+"): 0.01,
    ("10-13%", "10-13%"): 0.99,
    ("10-13%", "7-9%"): 0.3,
    ("10-13%", "14%+"): 0.3,
    ("14%+", "14%+"): 0.99,
    ("14%+", "10-13%"): 0.3,
    ("14%+", "7-9%"): 0.01,
}

def get_similarity_score(actual: str, expected: str, similarity_map: dict) -> float:
    """Get semantic similarity score between actual and expected values."""
    try:
        actual_clean = actual.lower().strip()
        expected_clean = expected.lower().strip()
        score = similarity_map.get((actual_clean, expected_clean), 0.01)
        return max(0.01, min(0.99, score))
    except Exception:
        return 0.01

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


def grade_risk_level(predicted: RiskLevel, expected: RiskLevel) -> float:
    """Score risk level: 1.0 exact, 0.3 adjacent, 0.01 two-levels off."""
    if predicted is None or expected is None:
        return 0.01
    if predicted not in RISK_LEVEL_ORDER or expected not in RISK_LEVEL_ORDER:
        return 0.01

    pred_ord = RISK_LEVEL_ORDER[predicted]
    exp_ord = RISK_LEVEL_ORDER[expected]
    distance = abs(pred_ord - exp_ord)

    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.3
    else:
        return 0.01


def grade_loan_decision(predicted: LoanDecision, expected: LoanDecision) -> float:
    """Score loan decision: 1.0 exact, 0.35 adjacent, 0.01 two-steps off."""
    if predicted is None or expected is None:
        return 0.01
    if predicted not in LOAN_DECISION_ORDER or expected not in LOAN_DECISION_ORDER:
        return 0.01

    pred_ord = LOAN_DECISION_ORDER[predicted]
    exp_ord = LOAN_DECISION_ORDER[expected]
    distance = abs(pred_ord - exp_ord)

    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.35
    else:
        return 0.01


def grade_interest_rate(predicted: InterestRateTier, expected: InterestRateTier) -> float:
    """Score interest rate tier: 1.0 exact, 0.3 adjacent, 0.01 two-tiers off."""
    if predicted is None or expected is None:
        return 0.01
    if predicted not in INTEREST_RATE_ORDER or expected not in INTEREST_RATE_ORDER:
        return 0.01

    pred_ord = INTEREST_RATE_ORDER[predicted]
    exp_ord = INTEREST_RATE_ORDER[expected]
    distance = abs(pred_ord - exp_ord)

    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.3
    else:
        return 0.01


def grade_consistency(action: Action) -> float:
    """
    Bonus/penalty in [-0.2, 0.15] for logical alignment across the three decisions.
    Contradictions are penalised based on 'Financial Integrity' standards.
    """
    if action is None:
        return 0.01

    risk = action.risk_level
    decision = action.loan_decision
    rate = action.interest_rate_tier

    if risk is None or decision is None or rate is None:
        return 0.01

    consistency_score = 0.0

    if risk == RiskLevel.LOW:
        if decision == LoanDecision.APPROVE:
            consistency_score += 0.05
        elif decision == LoanDecision.REJECT:
            consistency_score -= 0.05
        if rate == InterestRateTier.LOW:
            consistency_score += 0.05
        elif rate == InterestRateTier.HIGH:
            consistency_score -= 0.05

    elif risk == RiskLevel.MEDIUM:
        if decision == LoanDecision.CONDITIONAL_APPROVE:
            consistency_score += 0.05
        elif decision == LoanDecision.APPROVE and rate == InterestRateTier.LOW:
            consistency_score -= 0.05
        if rate == InterestRateTier.MEDIUM:
            consistency_score += 0.05

    elif risk == RiskLevel.HIGH:
        if decision in (LoanDecision.REJECT, LoanDecision.CONDITIONAL_APPROVE):
            consistency_score += 0.05
        elif decision == LoanDecision.APPROVE:
            consistency_score -= 0.10 # Severe optimism penalty
        if rate == InterestRateTier.HIGH:
            consistency_score += 0.05
        elif rate == InterestRateTier.LOW:
            consistency_score -= 0.20 # The "Irrational Pricing" Penalty

    # --- NEW: FINANCIAL INTEGRITY CHECKS (The Winning Tip: Quality of Envs) ---
    
    # Penalty: "The Irrational Pricing Penalty"
    # Giving a Low Rate to a High Risk applicant is a banking catastrophe.
    if risk == RiskLevel.HIGH and rate == InterestRateTier.LOW:
        consistency_score -= 0.20  # Severe penalty
    
    # Penalty: "The Optimism Trap"
    # Approving a High Risk applicant without high interest rates to offset risk.
    if risk == RiskLevel.HIGH and decision == LoanDecision.APPROVE and rate != InterestRateTier.HIGH:
        consistency_score -= 0.10

    # Bonus: "The Gold Standard Alignment"
    # Perfect alignment of Low Risk, Approve, and Low Rate.
    if risk == RiskLevel.LOW and decision == LoanDecision.APPROVE and rate == InterestRateTier.LOW:
        consistency_score += 0.05

    # Bonus: "Risk-Adjusted Pricing"
    # High Risk correctly paired with High Rate.
    if risk == RiskLevel.HIGH and rate == InterestRateTier.HIGH:
        consistency_score += 0.05

    # Clamp to [-0.2, 0.15] to allow for bigger penalties than bonuses
    return max(-0.2, min(0.15, consistency_score))


def grade_action(action: Action, ground_truth: GroundTruth) -> GradingResult:
    """
    Grade a complete action against ground truth.
    Weights: Risk 0.40, Decision 0.35, Rate 0.25, Consistency ±0.10.
    Returns GradingResult with per-component scores and total in [0.01, 0.99].
    """
    if action is None or ground_truth is None:
        return GradingResult(
            risk_level_score=0.01,
            loan_decision_score=0.01,
            interest_rate_score=0.01,
            consistency_bonus=0.0,
            total_score=0.01,
            feedback="❌ No valid action or ground truth provided.",
        )

    risk_score = grade_risk_level(action.risk_level, ground_truth.risk_level)
    decision_score = grade_loan_decision(action.loan_decision, ground_truth.loan_decision)
    rate_score = grade_interest_rate(action.interest_rate_tier, ground_truth.interest_rate_tier)
    consistency = grade_consistency(action)

    weighted_total = (
        risk_score * 0.40 +
        decision_score * 0.35 +
        rate_score * 0.25 +
        consistency
    )

    total_score = max(0.01, min(0.99, weighted_total))

    feedback_parts = []

    if risk_score >= 0.95:
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

    if decision_score >= 0.95:
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

    if rate_score >= 0.95:
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

    if consistency > 0:
        feedback_parts.append(f"🔗 Consistency Bonus: +{consistency:.2f} (Logical alignment confirmed)")
    elif consistency < 0:
        severity = "CRITICAL" if consistency <= -0.15 else "WARNING"
        feedback_parts.append(f"⚠️ {severity} Consistency Penalty: {consistency:.2f} (Irrational/Contradictory logic detected)")

    feedback = "📊 UNDERWRITING QUALITY AUDIT:\n" + "\n".join(feedback_parts)

    return GradingResult(
        risk_level_score=risk_score,
        loan_decision_score=decision_score,
        interest_rate_score=rate_score,
        consistency_bonus=consistency,
        total_score=total_score,
        feedback=feedback,
    )


def grade_lead_qualification(action: Action, ground_truth: GroundTruth) -> GradingResult:
    """
    Stage 1 grader (Sales). Decision weight raised to 0.45 because qualification
    correctness matters more than rate tier at this early stage.
    """
    if action is None or ground_truth is None:
        return GradingResult(
            risk_level_score=0.01,
            loan_decision_score=0.01,
            interest_rate_score=0.01,
            consistency_bonus=0.0,
            total_score=0.01,
            feedback="❌ No valid action or ground truth provided for lead qualification.",
        )

    risk_score = grade_risk_level(action.risk_level, ground_truth.risk_level)
    decision_score = grade_loan_decision(action.loan_decision, ground_truth.loan_decision)
    rate_score = grade_interest_rate(action.interest_rate_tier, ground_truth.interest_rate_tier)
    consistency = grade_consistency(action)

    weighted_total = (
        risk_score * 0.35 +
        decision_score * 0.45 +
        rate_score * 0.20 +
        consistency
    )

    total_score = max(0.01, min(0.99, weighted_total))

    feedback_parts = []
    if risk_score >= 0.95:
        feedback_parts.append(f"✅ Lead strength: Correct ({action.risk_level.value})")
    elif risk_score > 0.1:
        feedback_parts.append(
            f"⚠️ Lead strength: Partially correct — assessed {action.risk_level.value}, "
            f"expected {ground_truth.risk_level.value}"
        )
    else:
        feedback_parts.append(
            f"❌ Lead strength: Incorrect — assessed {action.risk_level.value}, "
            f"expected {ground_truth.risk_level.value}"
        )

    if decision_score >= 0.95:
        feedback_parts.append(f"✅ Qualification: Correct ({action.loan_decision.value})")
    elif decision_score > 0.1:
        feedback_parts.append(
            f"⚠️ Qualification: Partially correct — decided {action.loan_decision.value}, "
            f"expected {ground_truth.loan_decision.value}"
        )
    else:
        feedback_parts.append(
            f"❌ Qualification: Incorrect — decided {action.loan_decision.value}, "
            f"expected {ground_truth.loan_decision.value}"
        )

    if rate_score >= 0.95:
        feedback_parts.append(f"✅ Processing priority: Correct ({action.interest_rate_tier.value})")
    elif rate_score > 0.1:
        feedback_parts.append(
            f"⚠️ Priority: Partially correct — set {action.interest_rate_tier.value}, "
            f"expected {ground_truth.interest_rate_tier.value}"
        )
    else:
        feedback_parts.append(
            f"❌ Priority: Incorrect — set {action.interest_rate_tier.value}, "
            f"expected {ground_truth.interest_rate_tier.value}"
        )

    if consistency > 0:
        feedback_parts.append(f"🔗 Consistency bonus: +{consistency:.2f}")
    elif consistency < 0:
        feedback_parts.append(f"⚠️ Consistency penalty: {consistency:.2f}")

    feedback = "\n".join(feedback_parts)

    return GradingResult(
        risk_level_score=risk_score,
        loan_decision_score=decision_score,
        interest_rate_score=rate_score,
        consistency_bonus=consistency,
        total_score=total_score,
        feedback=feedback,
    )


def grade_document_verification(action: Action, ground_truth: GroundTruth) -> GradingResult:
    """
    Stage 2 grader (Document Verification). Risk weight raised to 0.45 because
    assessing document completeness is the primary objective at this stage.
    """
    if action is None or ground_truth is None:
        return GradingResult(
            risk_level_score=0.01,
            loan_decision_score=0.01,
            interest_rate_score=0.01,
            consistency_bonus=0.0,
            total_score=0.01,
            feedback="❌ No valid action or ground truth provided for document verification.",
        )

    risk_score = grade_risk_level(action.risk_level, ground_truth.risk_level)
    decision_score = grade_loan_decision(action.loan_decision, ground_truth.loan_decision)
    rate_score = grade_interest_rate(action.interest_rate_tier, ground_truth.interest_rate_tier)
    consistency = grade_consistency(action)

    weighted_total = (
        risk_score * 0.45 +
        decision_score * 0.35 +
        rate_score * 0.20 +
        consistency
    )

    total_score = max(0.01, min(0.99, weighted_total))

    feedback_parts = []
    if risk_score >= 0.95:
        feedback_parts.append(f"✅ Document assessment: Correct ({action.risk_level.value})")
    elif risk_score > 0.1:
        feedback_parts.append(
            f"⚠️ Document assessment: Partially correct — assessed {action.risk_level.value}, "
            f"expected {ground_truth.risk_level.value}"
        )
    else:
        feedback_parts.append(
            f"❌ Document assessment: Incorrect — assessed {action.risk_level.value}, "
            f"expected {ground_truth.risk_level.value}"
        )

    if decision_score >= 0.95:
        feedback_parts.append(f"✅ Verification decision: Correct ({action.loan_decision.value})")
    elif decision_score > 0.1:
        feedback_parts.append(
            f"⚠️ Verification: Partially correct — decided {action.loan_decision.value}, "
            f"expected {ground_truth.loan_decision.value}"
        )
    else:
        feedback_parts.append(
            f"❌ Verification: Incorrect — decided {action.loan_decision.value}, "
            f"expected {ground_truth.loan_decision.value}"
        )

    if rate_score >= 0.95:
        feedback_parts.append(f"✅ Processing tier: Correct ({action.interest_rate_tier.value})")
    elif rate_score > 0.1:
        feedback_parts.append(
            f"⚠️ Processing tier: Partially correct — set {action.interest_rate_tier.value}, "
            f"expected {ground_truth.interest_rate_tier.value}"
        )
    else:
        feedback_parts.append(
            f"❌ Processing tier: Incorrect — set {action.interest_rate_tier.value}, "
            f"expected {ground_truth.interest_rate_tier.value}"
        )

    if consistency > 0:
        feedback_parts.append(f"🔗 Consistency bonus: +{consistency:.2f}")
    elif consistency < 0:
        feedback_parts.append(f"⚠️ Consistency penalty: {consistency:.2f}")

    feedback = "\n".join(feedback_parts)

    return GradingResult(
        risk_level_score=risk_score,
        loan_decision_score=decision_score,
        interest_rate_score=rate_score,
        consistency_bonus=consistency,
        total_score=total_score,
        feedback=feedback,
    )


def grade_customer_onboarding(action: Action, ground_truth: GroundTruth) -> GradingResult:
    """
    Stage 5 grader (Customer Onboarding). Risk and decision weighted equally (0.35 each).
    +0.05 completeness bonus applied when all three components are exact matches.
    """
    if action is None or ground_truth is None:
        return GradingResult(
            risk_level_score=0.01,
            loan_decision_score=0.01,
            interest_rate_score=0.01,
            consistency_bonus=0.0,
            total_score=0.01,
            feedback="❌ No valid action or ground truth provided for customer onboarding.",
        )

    risk_score = grade_risk_level(action.risk_level, ground_truth.risk_level)
    decision_score = grade_loan_decision(action.loan_decision, ground_truth.loan_decision)
    rate_score = grade_interest_rate(action.interest_rate_tier, ground_truth.interest_rate_tier)
    consistency = grade_consistency(action)

    weighted_total = (
        risk_score * 0.35 +
        decision_score * 0.35 +
        rate_score * 0.20 +
        consistency
    )

    if risk_score >= 0.95 and decision_score >= 0.95 and rate_score >= 0.95:
        weighted_total += 0.05

    total_score = max(0.01, min(0.99, weighted_total))

    feedback_parts = []
    if risk_score >= 0.95:
        feedback_parts.append(f"✅ Onboarding readiness: Correct ({action.risk_level.value})")
    elif risk_score > 0.1:
        feedback_parts.append(
            f"⚠️ Onboarding readiness: Partially correct — assessed {action.risk_level.value}, "
            f"expected {ground_truth.risk_level.value}"
        )
    else:
        feedback_parts.append(
            f"❌ Onboarding readiness: Incorrect — assessed {action.risk_level.value}, "
            f"expected {ground_truth.risk_level.value}"
        )

    if decision_score >= 0.95:
        feedback_parts.append(f"✅ Disbursement decision: Correct ({action.loan_decision.value})")
    elif decision_score > 0.1:
        feedback_parts.append(
            f"⚠️ Disbursement: Partially correct — decided {action.loan_decision.value}, "
            f"expected {ground_truth.loan_decision.value}"
        )
    else:
        feedback_parts.append(
            f"❌ Disbursement: Incorrect — decided {action.loan_decision.value}, "
            f"expected {ground_truth.loan_decision.value}"
        )

    if rate_score >= 0.95:
        feedback_parts.append(f"✅ Disbursement priority: Correct ({action.interest_rate_tier.value})")
    elif rate_score > 0.1:
        feedback_parts.append(
            f"⚠️ Priority: Partially correct — set {action.interest_rate_tier.value}, "
            f"expected {ground_truth.interest_rate_tier.value}"
        )
    else:
        feedback_parts.append(
            f"❌ Priority: Incorrect — set {action.interest_rate_tier.value}, "
            f"expected {ground_truth.interest_rate_tier.value}"
        )

    if consistency > 0:
        feedback_parts.append(f"🔗 Consistency bonus: +{consistency:.2f}")
    elif consistency < 0:
        feedback_parts.append(f"⚠️ Consistency penalty: {consistency:.2f}")

    if risk_score >= 0.95 and decision_score >= 0.95 and rate_score >= 0.95:
        feedback_parts.append("🎯 All onboarding steps assessed correctly — completeness bonus applied!")

    feedback = "\n".join(feedback_parts)

    return GradingResult(
        risk_level_score=risk_score,
        loan_decision_score=decision_score,
        interest_rate_score=rate_score,
        consistency_bonus=consistency,
        total_score=total_score,
        feedback=feedback,
    )

def calculate_dynamic_ground_truth(obs: "ApplicantProfile") -> GroundTruth:
    """Derive ground truth from applicant data for custom profiles."""
    score = obs.credit_score
    income = obs.annual_income
    debt = obs.existing_debt
    loan = obs.loan_amount_requested
    defaults = getattr(obs, 'previous_defaults', 0)
    
    dti = (debt / income) if income > 0 else 1.0
    lti = (loan / income) if income > 0 else 2.0
    
    risk_level = RiskLevel.MEDIUM
    if score >= 740 and dti < 0.35 and defaults == 0:
        risk_level = RiskLevel.LOW
    elif score < 620 or dti > 0.60 or defaults > 1:
        risk_level = RiskLevel.HIGH

    loan_decision = LoanDecision.CONDITIONAL_APPROVE
    if risk_level == RiskLevel.LOW and lti < 3.0:
        loan_decision = LoanDecision.APPROVE
    elif risk_level == RiskLevel.HIGH or lti > 5.0:
        loan_decision = LoanDecision.REJECT

    interest_rate = InterestRateTier.MEDIUM
    if risk_level == RiskLevel.LOW:
        interest_rate = InterestRateTier.LOW
    elif risk_level == RiskLevel.HIGH:
        interest_rate = InterestRateTier.HIGH

    explanation = get_underwriting_explanation(obs, risk_level, loan_decision, interest_rate)
    
    return GroundTruth(
        risk_level=risk_level,
        loan_decision=loan_decision,
        interest_rate_tier=interest_rate,
        explanation=explanation
    )

def get_underwriting_explanation(obs: "ApplicantProfile", risk: RiskLevel, dec: LoanDecision, rate: InterestRateTier) -> str:
    """Generate a detailed explanation for the dynamic ground truth following the 5-stage process."""
    dti = (obs.existing_debt / obs.annual_income * 100) if obs.annual_income > 0 else 0
    collateral_str = "Provided" if getattr(obs, 'has_collateral', False) else "None"
    
    docs = getattr(obs, 'documents_submitted', [])
    doc_status = f"Verified ({', '.join(docs)})" if docs else "Pending / Missing critical files"
    
    parts = [
        f"AUTONOMOUS UNDERWRITING REPORT FOR {obs.applicant_name}:",
        "",
        "STAGE 1: DOCUMENTATION & IDENTITY VERIFICATION",
        f"   - Status: {doc_status}. Identity and income sources confirmed.",
        "",
        "STAGE 2: CREDIT CHARACTER ASSESSMENT",
        f"   - Analysis: Credit score of {obs.credit_score} indicates {'strong' if obs.credit_score > 740 else 'moderate' if obs.credit_score > 650 else 'high-risk'} character. Previous defaults: {getattr(obs, 'previous_defaults', 0)}.",
        "",
        "STAGE 3: CAPACITY & CAPITAL ANALYSIS",
        f"   - Assessment: Debt-to-Income (DTI) ratio is {dti:.1f}%. Capital profile (Employment: {obs.employment_type.value}, Tenure: {getattr(obs, 'employment_years', 0)} yrs).",
        "",
        "STAGE 4: COLLATERAL & CONDITIONS",
        f"   - Status: Collateral ({collateral_str}). Tenure: {getattr(obs, 'repayment_tenure_months', 36)} months. Interest rate target: {rate.value}.",
        "",
        "STAGE 5: FINAL UNDERWRITING DECISION",
        f"   - Synthesized Risk: {risk.value.upper()}.",
        f"   - FINAL VERDICT: {dec.value.upper()}."
    ]
    
    return "\n".join(parts)
