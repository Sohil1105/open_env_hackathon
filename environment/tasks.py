"""
Task definitions for the Loan Underwriting OpenEnv environment.

Defines 3 tasks with progressive difficulty:
- Easy: Salaried employee, high credit score, low debt-to-income
- Medium: Self-employed, average credit score, moderate debt
- Hard: Freelancer, low credit score, high debt, large loan request

Each task includes a full applicant profile and its ground truth answers.
"""

from .models import (
    ApplicantProfile,
    GroundTruth,
    EmploymentType,
    RiskLevel,
    LoanDecision,
    InterestRateTier,
    TaskDifficulty,
)


# ─── Task Data Structure ─────────────────────────────────────────────────────

class TaskDefinition:
    """
    Encapsulates a complete task: the applicant profile, the expected
    ground truth, and metadata about what the agent must do.
    """

    def __init__(
        self,
        task_id: str,
        name: str,
        difficulty: TaskDifficulty,
        description: str,
        profile: ApplicantProfile,
        ground_truth: GroundTruth,
    ):
        self.task_id = task_id
        self.name = name
        self.difficulty = difficulty
        self.description = description
        self.profile = profile
        self.ground_truth = ground_truth


# ─── Task 1: Easy — Salaried, High Credit, Low Risk ─────────────────────────

EASY_TASK = TaskDefinition(
    task_id="easy_salaried_high_credit",
    name="Simple Salaried Profile Assessment",
    difficulty=TaskDifficulty.EASY,
    description=(
        "You are evaluating a loan application from a salaried employee with an excellent "
        "financial profile. The applicant has a high credit score (above 750), a stable "
        "employment history, and a low debt-to-income ratio. Based on the applicant's "
        "profile, you must:\n"
        "1. Classify the applicant's risk level (Low / Medium / High)\n"
        "2. Make a loan decision (Approve / Conditional Approve / Reject)\n"
        "3. Recommend an interest rate tier (7-9% / 10-13% / 14%+)\n\n"
        "Focus primarily on correctly assessing the risk level. Given the strong profile, "
        "the loan decision and interest rate should follow naturally."
    ),
    profile=ApplicantProfile(
        applicant_name="Rajesh Kumar Sharma",
        age=35,
        annual_income=1_200_000.0,       # ₹1.2M annual income
        credit_score=785,                 # Excellent credit
        existing_debt=150_000.0,          # Low relative debt
        employment_type=EmploymentType.SALARIED,
        employment_years=8.5,             # Stable employment
        loan_amount_requested=500_000.0,  # Reasonable loan amount
        repayment_tenure_months=60,       # 5-year repayment
        monthly_expenses=45_000.0,        # Manageable expenses
        has_collateral=True,              # Has collateral
        previous_defaults=0,             # No defaults
    ),
    ground_truth=GroundTruth(
        risk_level=RiskLevel.LOW,
        loan_decision=LoanDecision.APPROVE,
        interest_rate_tier=InterestRateTier.LOW,
        explanation=(
            "This applicant has an excellent profile: high credit score (785), salaried "
            "employment with 8.5 years tenure, very low debt-to-income ratio (~12.5%), "
            "the loan amount is well within affordability at ~42% of annual income, "
            "they have collateral, and zero previous defaults. This is a textbook low-risk "
            "applicant who should be approved at the best interest rate tier (7-9%)."
        ),
    ),
)


# ─── Task 2: Medium — Self-Employed, Average Credit, Moderate Risk ───────────

MEDIUM_TASK = TaskDefinition(
    task_id="medium_self_employed_moderate",
    name="Self-Employed Moderate Profile",
    difficulty=TaskDifficulty.MEDIUM,
    description=(
        "You are evaluating a loan application from a self-employed business owner. The "
        "applicant has an average credit score (620–700), moderate existing debt, and "
        "variable income stability. This is a nuanced case requiring careful analysis.\n\n"
        "Based on the applicant's profile, you must:\n"
        "1. Classify the applicant's risk level (Low / Medium / High)\n"
        "2. Make a loan decision (Approve / Conditional Approve / Reject)\n"
        "3. Recommend an interest rate tier (7-9% / 10-13% / 14%+)\n\n"
        "Pay close attention to the debt-to-income ratio, employment stability, and "
        "credit history. The agent must balance multiple competing signals to arrive at "
        "a well-reasoned decision."
    ),
    profile=ApplicantProfile(
        applicant_name="Priya Venkatesh",
        age=42,
        annual_income=720_000.0,          # Decent but variable income
        credit_score=665,                  # Average credit
        existing_debt=280_000.0,           # Moderate existing debt
        employment_type=EmploymentType.SELF_EMPLOYED,
        employment_years=5.0,              # Moderate tenure
        loan_amount_requested=400_000.0,   # Moderate loan request
        repayment_tenure_months=84,        # 7-year repayment
        monthly_expenses=38_000.0,         # Significant expenses
        has_collateral=True,               # Has collateral (mitigating factor)
        previous_defaults=1,              # One previous default (red flag)
    ),
    ground_truth=GroundTruth(
        risk_level=RiskLevel.MEDIUM,
        loan_decision=LoanDecision.CONDITIONAL_APPROVE,
        interest_rate_tier=InterestRateTier.MEDIUM,
        explanation=(
            "This applicant presents a mixed profile: average credit score (665) with one "
            "previous default, self-employed with variable income stability, debt-to-income "
            "ratio of ~38.9% (moderate-high), and requesting ~55.6% of annual income as a loan. "
            "However, mitigating factors include 5 years of business operation, available "
            "collateral, and a 7-year repayment window. The appropriate decision is conditional "
            "approval at a medium interest rate (10-13%) with conditions such as proof of "
            "consistent business income and documentation of the previous default resolution."
        ),
    ),
)


# ─── Task 3: Hard — Freelancer, Low Credit, High Risk ────────────────────────

HARD_TASK = TaskDefinition(
    task_id="hard_freelancer_complex",
    name="Complex Freelancer High-Risk Profile",
    difficulty=TaskDifficulty.HARD,
    description=(
        "You are evaluating a complex loan application from a freelancer with irregular "
        "income patterns. The applicant has a low credit score (550–600), high existing "
        "debt, and is requesting a large loan. This is a high-risk case that requires "
        "careful analysis of all factors.\n\n"
        "Based on the applicant's profile, you must:\n"
        "1. Classify the applicant's risk level (Low / Medium / High)\n"
        "2. Make a loan decision (Approve / Conditional Approve / Reject)\n"
        "3. Recommend an interest rate tier (7-9% / 10-13% / 14%+)\n\n"
        "IMPORTANT: All three decisions must align logically. A high-risk classification "
        "should correspond with either rejection or conditional approval at a high interest "
        "rate. Consider the debt-to-income ratio, credit history, income stability, and "
        "the size of the loan request relative to the applicant's financial capacity."
    ),
    profile=ApplicantProfile(
        applicant_name="Arjun Mehta",
        age=29,
        annual_income=420_000.0,           # Lower, irregular income
        credit_score=572,                   # Poor credit score
        existing_debt=380_000.0,            # Very high existing debt
        employment_type=EmploymentType.FREELANCER,
        employment_years=2.5,               # Short freelancing history
        loan_amount_requested=650_000.0,    # Large loan (155% of annual income!)
        repayment_tenure_months=120,        # 10-year repayment
        monthly_expenses=32_000.0,          # Expenses eating into income
        has_collateral=False,               # No collateral
        previous_defaults=2,               # Multiple previous defaults
    ),
    ground_truth=GroundTruth(
        risk_level=RiskLevel.HIGH,
        loan_decision=LoanDecision.REJECT,
        interest_rate_tier=InterestRateTier.HIGH,
        explanation=(
            "This applicant is clearly high-risk: credit score of 572 is poor, debt-to-income "
            "ratio is ~90.5% (extremely high), requesting a loan that is 155% of annual income "
            "with no collateral, freelancer with only 2.5 years of irregular income history, "
            "and has 2 previous defaults. Monthly income (~₹35,000) minus expenses (₹32,000) "
            "leaves only ₹3,000/month — insufficient to service existing debt plus a new large "
            "loan. The application should be rejected. If any rate were offered, it would need "
            "to be in the highest tier (14%+) to compensate for the extreme risk."
        ),
    ),
)


# ─── Task 4: Bankruptcy Recovery ───────────────────────────────────────────────

BANKRUPTCY_TASK = TaskDefinition(
    task_id="bankruptcy_recovery_edge1",
    name="Bankruptcy Recovery Edge Case",
    difficulty=TaskDifficulty.MEDIUM,
    description=(
        "You are evaluating a loan application from an individual who declared bankruptcy "
        "7 years ago. They have since rebuilt their credit.\n\n"
        "Based on the applicant's profile, you must:\n"
        "1. Classify the applicant's risk level (Low / Medium / High)\n"
        "2. Make a loan decision (Approve / Conditional Approve / Reject)\n"
        "3. Recommend an interest rate tier (7-9% / 10-13% / 14%+)\n"
    ),
    profile=ApplicantProfile(
        applicant_name="John Doe",
        age=45,
        annual_income=65_000.0,
        credit_score=680,
        existing_debt=8_000.0,
        employment_type=EmploymentType.SALARIED,
        employment_years=4.0,
        loan_amount_requested=120_000.0,
        repayment_tenure_months=120,
        monthly_expenses=2_500.0,
        has_collateral=False,
        previous_defaults=1,
    ),
    ground_truth=GroundTruth(
        risk_level=RiskLevel.MEDIUM,
        loan_decision=LoanDecision.CONDITIONAL_APPROVE,
        interest_rate_tier=InterestRateTier.MEDIUM,
        explanation=(
            "This applicant has rebuilt their credit (680) after a bankruptcy 7 years ago. "
            "Their current debt is manageable (₹8,000) against an income of ₹65,000, but the "
            "loan amount is relatively high (₹120,000) with no collateral. A medium risk "
            "classification with a conditional approval and a 10-13% interest rate is appropriate."
        ),
    ),
)


# ─── Task 5: Joint Applicants ────────────────────────────────────────────────

JOINT_APP_TASK = TaskDefinition(
    task_id="joint_applicants_edge2",
    name="Joint Applicants Edge Case",
    difficulty=TaskDifficulty.EASY,
    description=(
        "You are evaluating a loan application from joint applicants with combined income.\n\n"
        "Based on the applicant's profile, you must:\n"
        "1. Classify the applicant's risk level (Low / Medium / High)\n"
        "2. Make a loan decision (Approve / Conditional Approve / Reject)\n"
        "3. Recommend an interest rate tier (7-9% / 10-13% / 14%+)\n"
    ),
    profile=ApplicantProfile(
        applicant_name="The Smiths",
        age=38,
        annual_income=120_000.0,
        credit_score=720,
        existing_debt=25_000.0,
        employment_type=EmploymentType.SALARIED,
        employment_years=6.0,
        loan_amount_requested=300_000.0,
        repayment_tenure_months=360,
        monthly_expenses=4_000.0,
        has_collateral=True,
        previous_defaults=0,
    ),
    ground_truth=GroundTruth(
        risk_level=RiskLevel.LOW,
        loan_decision=LoanDecision.APPROVE,
        interest_rate_tier=InterestRateTier.LOW,
        explanation=(
            "The joint applicants have a strong combined income of ₹120,000. "
            "The primary applicant has a good credit score (720). The existing "
            "debt of ₹25,000 is manageable, and they provide collateral for "
            "the ₹300,000 loan. This results in a low risk classification, "
            "an approval, and a 7-9% interest rate."
        ),
    ),
)


# ─── Task 6: Lead Qualification (Sales) ──────────────────────────────────────

LEAD_QUALIFICATION_TASK = TaskDefinition(
    task_id="lead_qualification_sales",
    name="Lead Qualification Assessment",
    difficulty=TaskDifficulty.EASY,
    description=(
        "You are a sales team member performing initial lead qualification on a potential "
        "loan applicant. Based on their initial inquiry details, you must decide whether "
        "this lead is worthy of a full loan application.\n\n"
        "Based on the applicant's initial profile, you must:\n"
        "1. Classify the lead strength (Low Risk = Strong Lead / Medium = Borderline / High = Weak Lead)\n"
        "2. Make a qualification decision (Approve = Qualify for Full Application / "
        "Conditional Approve = Request More Info / Reject = Disqualify)\n"
        "3. Recommend a processing priority tier (7-9% = High Priority / 10-13% = Standard / 14%+ = Low Priority)\n\n"
        "Key decision factors:\n"
        "- High income + stable employment = Qualify immediately (score high)\n"
        "- Low income + unstable employment = Disqualify (score high for correct rejection)\n"
        "- Borderline cases = Request more information (partial credit)"
    ),
    profile=ApplicantProfile(
        applicant_name="Anika Patel",
        age=31,
        annual_income=950_000.0,         # High income
        credit_score=760,                 # Excellent credit
        existing_debt=80_000.0,           # Low debt
        employment_type=EmploymentType.SALARIED,
        employment_years=7.0,             # Stable employment
        loan_amount_requested=350_000.0,  # Reasonable ask
        repayment_tenure_months=48,
        monthly_expenses=35_000.0,
        has_collateral=True,
        previous_defaults=0,
    ),
    ground_truth=GroundTruth(
        risk_level=RiskLevel.LOW,
        loan_decision=LoanDecision.APPROVE,
        interest_rate_tier=InterestRateTier.LOW,
        explanation=(
            "Strong lead: high annual income (₹950K), excellent credit score (760), "
            "salaried with 7 years tenure, low DTI ratio (~8.4%), reasonable loan request "
            "(~36.8% of income), has collateral, zero defaults. This lead should be "
            "immediately qualified and prioritized for full application processing."
        ),
    ),
)


# ─── Task 7: Document Verification (HR/IT) ──────────────────────────────────

DOCUMENT_VERIFICATION_TASK = TaskDefinition(
    task_id="document_verification_hr",
    name="Document Verification Assessment",
    difficulty=TaskDifficulty.MEDIUM,
    description=(
        "You are performing document verification for a loan application. You must assess "
        "whether the submitted documents are complete, consistent, and sufficient for "
        "loan processing.\n\n"
        "Based on the applicant's profile and document indicators, you must:\n"
        "1. Classify document risk (Low = Complete & Consistent / Medium = Missing Documents / "
        "High = Suspicious or Inconsistent)\n"
        "2. Make a verification decision (Approve = All Clear / Conditional Approve = "
        "Request Missing Docs / Reject = Flag Suspicious)\n"
        "3. Recommend processing tier (7-9% = Fast Track / 10-13% = Standard Review / "
        "14%+ = Enhanced Scrutiny)\n\n"
        "Key indicators: Self-employed applicants with short tenure and previous defaults "
        "suggest potential document inconsistencies. Missing collateral documentation and "
        "moderate debt levels require additional verification."
    ),
    profile=ApplicantProfile(
        applicant_name="Vikram Desai",
        age=37,
        annual_income=580_000.0,          # Decent income
        credit_score=640,                  # Fair credit
        existing_debt=220_000.0,           # Moderate-high debt
        employment_type=EmploymentType.SELF_EMPLOYED,
        employment_years=3.5,              # Short tenure
        loan_amount_requested=450_000.0,
        repayment_tenure_months=72,
        monthly_expenses=30_000.0,
        has_collateral=False,              # No collateral docs
        previous_defaults=1,              # Past default
    ),
    ground_truth=GroundTruth(
        risk_level=RiskLevel.MEDIUM,
        loan_decision=LoanDecision.CONDITIONAL_APPROVE,
        interest_rate_tier=InterestRateTier.MEDIUM,
        explanation=(
            "Document verification reveals concerns: self-employed with only 3.5 years tenure "
            "means income documentation may be incomplete. One previous default requires "
            "resolution documentation. No collateral means property documents are absent. "
            "DTI of ~37.9% is moderate-high. Documents should be conditionally accepted with "
            "requests for: business income proof (3 years), default resolution letter, and "
            "additional identity verification. Standard 10-13% processing tier."
        ),
    ),
)


# ─── Task 8: Customer Onboarding (Project Management) ───────────────────────

CUSTOMER_ONBOARDING_TASK = TaskDefinition(
    task_id="customer_onboarding_pm",
    name="Customer Onboarding Assessment",
    difficulty=TaskDifficulty.EASY,
    description=(
        "You are a project manager handling the onboarding of an approved loan customer. "
        "The loan has been approved and you must assess the completeness of the onboarding "
        "process and readiness for loan disbursement.\n\n"
        "Based on the approved loan details, you must:\n"
        "1. Classify onboarding readiness (Low Risk = All Steps Complete / Medium = "
        "Steps Missing / High = Critical Gaps)\n"
        "2. Make an onboarding decision (Approve = Ready to Disburse / Conditional Approve = "
        "Pending Steps / Reject = Restart Onboarding)\n"
        "3. Recommend disbursement priority (7-9% = Immediate Disbursement / 10-13% = "
        "Within 7 Days / 14%+ = Delayed)\n\n"
        "Onboarding checklist: KYC verification, agreement signing, account setup, "
        "insurance linkage, EMI mandate registration. All steps completed in order = "
        "full score. Steps skipped = partial score. Wrong order = penalty."
    ),
    profile=ApplicantProfile(
        applicant_name="Meera Joshi",
        age=40,
        annual_income=1_050_000.0,        # High income
        credit_score=745,                  # Good credit
        existing_debt=120_000.0,           # Manageable debt
        employment_type=EmploymentType.SALARIED,
        employment_years=12.0,             # Very stable
        loan_amount_requested=600_000.0,
        repayment_tenure_months=60,
        monthly_expenses=42_000.0,
        has_collateral=True,
        previous_defaults=0,
    ),
    ground_truth=GroundTruth(
        risk_level=RiskLevel.LOW,
        loan_decision=LoanDecision.APPROVE,
        interest_rate_tier=InterestRateTier.LOW,
        explanation=(
            "Customer onboarding assessment: All steps are complete. Excellent credentials: "
            "high income ($1.05M), good credit (745), 12 years stable salaried employment, "
            "low DTI (~11.4%), collateral documented, zero defaults. All onboarding steps "
            "(KYC, agreement signing, account setup, insurance linkage, EMI mandate) can be "
            "completed in order. Ready for immediate loan disbursement at best priority tier."
        ),
    ),
)


# ─── Registry of all tasks ───────────────────────────────────────────────────

ALL_TASKS = {
    "lead_qualification_sales": LEAD_QUALIFICATION_TASK,
    "document_verification_hr": DOCUMENT_VERIFICATION_TASK,
    "easy_salaried_high_credit": EASY_TASK,
    "medium_self_employed_moderate": MEDIUM_TASK,
    "hard_freelancer_complex": HARD_TASK,
    "customer_onboarding_pm": CUSTOMER_ONBOARDING_TASK,
    "bankruptcy_recovery_edge1": BANKRUPTCY_TASK,
    "joint_applicants_edge2": JOINT_APP_TASK,
}

# Ordered list for sequential execution (loan lifecycle order)
TASK_ORDER = [
    "lead_qualification_sales",        # Stage 1 — Lead Qualification (Sales)
    "document_verification_hr",        # Stage 2 — Document Verification (HR/IT)
    "easy_salaried_high_credit",       # Stage 3 — Risk Assessment (Easy)
    "medium_self_employed_moderate",   # Stage 4 — Risk Assessment (Medium)
    "hard_freelancer_complex",         # Stage 5 — Risk Assessment (Hard)
    "customer_onboarding_pm",          # Stage 6 — Customer Onboarding (PM)
    "bankruptcy_recovery_edge1",       # Stage 7 — Portfolio Monitoring
    "joint_applicants_edge2",          # Stage 8 — Loan Closure
]

def get_task(task_id: str) -> TaskDefinition:
    """Retrieve a task by its ID. Raises KeyError if not found."""
    if task_id not in ALL_TASKS:
        raise KeyError(
            f"Unknown task ID: '{task_id}'. "
            f"Available tasks: {list(ALL_TASKS.keys())}"
        )
    return ALL_TASKS[task_id]



def get_all_tasks() -> list[TaskDefinition]:
    """Return all tasks in order of difficulty."""
    return [ALL_TASKS[tid] for tid in TASK_ORDER]


def generate_heuristic_ground_truth(profile: ApplicantProfile) -> GroundTruth:
    """
    Generate a logical GroundTruth for a given ApplicantProfile using heuristics.
    Used for custom user-submitted profiles.
    """
    # 1. Determine Risk Level
    dti = profile.debt_to_income_ratio
    
    if profile.credit_score >= 740 and dti < 0.25 and profile.previous_defaults == 0:
        risk_level = RiskLevel.LOW
    elif profile.credit_score < 600 or dti > 0.55 or profile.previous_defaults >= 2:
        risk_level = RiskLevel.HIGH
    else:
        risk_level = RiskLevel.MEDIUM

    # 2. Determine Loan Decision
    if risk_level == RiskLevel.LOW:
        loan_decision = LoanDecision.APPROVE
    elif risk_level == RiskLevel.MEDIUM:
        if profile.has_collateral or dti < 0.35:
            loan_decision = LoanDecision.APPROVE
        else:
            loan_decision = LoanDecision.CONDITIONAL_APPROVE
    else:  # High Risk
        if profile.has_collateral and dti < 0.45 and profile.credit_score > 550:
            loan_decision = LoanDecision.CONDITIONAL_APPROVE
        else:
            loan_decision = LoanDecision.REJECT

    # 3. Determine Interest Rate Tier
    if risk_level == RiskLevel.LOW:
        interest_tier = InterestRateTier.LOW
    elif risk_level == RiskLevel.MEDIUM:
        interest_tier = InterestRateTier.MEDIUM
    else:
        interest_tier = InterestRateTier.HIGH

    # 4. Generate Explanation
    explanation = (
        f"Heuristic Evaluation: Risk is {risk_level.value} due to credit score {profile.credit_score} "
        f"and DTI {dti:.2%}. Decision is {loan_decision.value} "
        f"{'with collateral' if profile.has_collateral else 'without collateral'}."
    )

    return GroundTruth(
        risk_level=risk_level,
        loan_decision=loan_decision,
        interest_rate_tier=interest_tier,
        explanation=explanation,
    )
