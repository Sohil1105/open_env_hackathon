"""
Pydantic typed models for the Loan Underwriting OpenEnv environment.

Defines the core data structures:
- ApplicantProfile: Raw applicant data loaded per episode
- Observation: What the agent sees (applicant profile + task context)
- Action: The agent's underwriting decision (risk, approval, rate)
- State: Full environment state including observation, ground truth, and scores
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ─── Enums for constrained fields ────────────────────────────────────────────

class EmploymentType(str, Enum):
    """Employment classification for the loan applicant."""
    SALARIED = "salaried"
    SELF_EMPLOYED = "self_employed"
    FREELANCER = "freelancer"
    CONTRACT = "contract"
    UNEMPLOYED = "unemployed"


class RiskLevel(str, Enum):
    """Risk classification tiers for loan applicants."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class LoanDecision(str, Enum):
    """Loan approval decision options."""
    APPROVE = "Approve"
    CONDITIONAL_APPROVE = "Conditional Approve"
    REJECT = "Reject"


class InterestRateTier(str, Enum):
    """Interest rate tier recommendations."""
    LOW = "7-9%"
    MEDIUM = "10-13%"
    HIGH = "14%+"


class TaskDifficulty(str, Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ─── Applicant Profile ───────────────────────────────────────────────────────

class ApplicantProfile(BaseModel):
    """
    Complete financial and personal profile of a loan applicant.
    This is the raw data that gets transformed into an Observation.
    """
    applicant_name: str = Field(..., description="Full name of the loan applicant")
    age: int = Field(..., ge=18, le=100, description="Age in years")
    annual_income: float = Field(..., gt=0, description="Annual income in USD")
    credit_score: int = Field(..., ge=300, le=850, description="FICO credit score")
    existing_debt: float = Field(..., ge=0, description="Total existing debt in USD")
    employment_type: EmploymentType = Field(..., description="Employment classification")
    employment_years: float = Field(..., ge=0, description="Years in current employment")
    loan_amount_requested: float = Field(..., gt=0, description="Requested loan amount in USD")
    repayment_tenure_months: int = Field(..., gt=0, le=360, description="Repayment period in months")
    monthly_expenses: float = Field(..., ge=0, description="Average monthly expenses in USD")
    has_collateral: bool = Field(..., description="Whether applicant offers collateral")
    previous_defaults: int = Field(..., ge=0, description="Number of previous loan defaults")
    documents_submitted: Optional[list[str]] = Field(default=[], description="List of documents provided by the applicant")

    @property
    def debt_to_income_ratio(self) -> float:
        """Calculate the debt-to-income ratio."""
        if self.annual_income == 0:
            return float("inf")
        return self.existing_debt / self.annual_income

    @property
    def monthly_income(self) -> float:
        """Calculate monthly income from annual income."""
        return self.annual_income / 12.0

    @property
    def loan_to_income_ratio(self) -> float:
        """Calculate the loan-to-income ratio."""
        if self.annual_income == 0:
            return float("inf")
        return self.loan_amount_requested / self.annual_income


# ─── Observation ──────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    What the agent sees at each step.
    Contains the applicant profile data plus the task description
    telling the agent what decisions it needs to make.
    """
    applicant_name: str
    age: int
    annual_income: float
    credit_score: int
    existing_debt: float
    employment_type: str
    employment_years: float
    loan_amount_requested: float
    repayment_tenure_months: int
    monthly_expenses: float
    has_collateral: bool
    previous_defaults: int
    documents_submitted: Optional[list[str]] = Field(default=[], description="Documents verified")
    task_description: str = Field(..., description="What the agent must decide")
    task_id: str = Field(..., description="Unique identifier for the current task")
    task_difficulty: str = Field(..., description="Difficulty level: easy, medium, hard")

    @classmethod
    def from_profile(
        cls,
        profile: ApplicantProfile,
        task_description: str,
        task_id: str,
        task_difficulty: str,
    ) -> "Observation":
        """Create an Observation from an ApplicantProfile and task metadata."""
        return cls(
            applicant_name=profile.applicant_name,
            age=profile.age,
            annual_income=profile.annual_income,
            credit_score=profile.credit_score,
            existing_debt=profile.existing_debt,
            employment_type=profile.employment_type.value,
            employment_years=profile.employment_years,
            loan_amount_requested=profile.loan_amount_requested,
            repayment_tenure_months=profile.repayment_tenure_months,
            monthly_expenses=profile.monthly_expenses,
            has_collateral=profile.has_collateral,
            previous_defaults=profile.previous_defaults,
            documents_submitted=profile.documents_submitted,
            task_description=task_description,
            task_id=task_id,
            task_difficulty=task_difficulty,
        )


# ─── Action ───────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """
    The agent's underwriting decision.
    Contains three key decisions plus optional reasoning.
    """
    risk_level: RiskLevel = Field(..., description="Classified risk level")
    loan_decision: LoanDecision = Field(..., description="Loan approval decision")
    interest_rate_tier: InterestRateTier = Field(..., description="Recommended interest rate tier")
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning for the decision (optional but encouraged)"
    )

    @field_validator("risk_level", mode="before")
    @classmethod
    def normalize_risk_level(cls, v):
        """Normalize risk level input to handle case variations."""
        if isinstance(v, str):
            mapping = {"low": "Low", "medium": "Medium", "high": "High"}
            return mapping.get(v.lower().strip(), v)
        return v

    @field_validator("loan_decision", mode="before")
    @classmethod
    def normalize_loan_decision(cls, v):
        """Normalize loan decision input to handle case variations."""
        if isinstance(v, str):
            v_lower = v.lower().strip()
            mapping = {
                "approve": "Approve",
                "conditional approve": "Conditional Approve",
                "conditional": "Conditional Approve",
                "reject": "Reject",
                "deny": "Reject",
            }
            return mapping.get(v_lower, v)
        return v

    @field_validator("interest_rate_tier", mode="before")
    @classmethod
    def normalize_interest_rate_tier(cls, v):
        """Normalize interest rate tier input."""
        if isinstance(v, str):
            v_clean = v.strip().replace(" ", "")
            mapping = {
                "7-9%": "7-9%",
                "7%-9%": "7-9%",
                "10-13%": "10-13%",
                "10%-13%": "10-13%",
                "14%+": "14%+",
                "14+%": "14%+",
                "14%plus": "14%+",
            }
            return mapping.get(v_clean, v)
        return v


# ─── Ground Truth ─────────────────────────────────────────────────────────────

class GroundTruth(BaseModel):
    """
    The correct answers for a given applicant profile.
    Used by graders to evaluate the agent's decisions.
    """
    risk_level: RiskLevel
    loan_decision: LoanDecision
    interest_rate_tier: InterestRateTier
    explanation: str = Field(..., description="Why these are the correct answers")


# ─── Grading Result ──────────────────────────────────────────────────────────

class GradingResult(BaseModel):
    """
    Detailed breakdown of the grading for a single step.
    """
    risk_level_score: float = Field(..., ge=0.0, le=1.0, description="Score for risk classification")
    loan_decision_score: float = Field(..., ge=0.0, le=1.0, description="Score for loan decision")
    interest_rate_score: float = Field(..., ge=0.0, le=1.0, description="Score for interest rate tier")
    consistency_bonus: float = Field(
        default=0.0, ge=-0.1, le=0.1,
        description="Bonus/penalty for logical consistency"
    )
    total_score: float = Field(..., ge=0.0, le=1.0, description="Final weighted score")
    feedback: str = Field(..., description="Human-readable feedback on the decision")


# ─── Environment State ────────────────────────────────────────────────────────

class State(BaseModel):
    """
    Full environment state returned by env.state().
    Contains the current observation, whether an action has been taken,
    and any grading results from the most recent step.
    """
    observation: Optional[Observation] = Field(
        default=None, description="Current observation (applicant profile + task)"
    )
    action_taken: Optional[Action] = Field(
        default=None, description="Last action taken by the agent"
    )
    grading_result: Optional[GradingResult] = Field(
        default=None, description="Grading result from the last step"
    )
    done: bool = Field(default=False, description="Whether the episode is finished")
    current_task_id: Optional[str] = Field(
        default=None, description="ID of the current task being evaluated"
    )
    episode_reward: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Cumulative reward for this episode"
    )
