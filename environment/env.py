"""
Main OpenEnv environment class for Financial Loan Underwriting & Risk Assessment.

Implements the core OpenEnv interface:
- reset(task_id) → loads a new applicant profile, returns initial state
- step(action) → evaluates the agent's underwriting decision, returns (state, reward, done, info)
- state() → returns current environment state

This environment simulates a bank's loan underwriting desk. Each episode
presents one applicant profile for the agent to evaluate.
"""

from typing import Optional

from .models import (
    ApplicantProfile,
    Observation,
    Action,
    State,
    GradingResult,
    TaskDifficulty,
)
from .tasks import TaskDefinition, get_task, get_all_tasks, TASK_ORDER, generate_heuristic_ground_truth
from .rewards import compute_reward, format_reward_breakdown


class LoanUnderwritingEnv:
    """
    OpenEnv-compliant environment for financial loan underwriting.

    Each episode:
    1. Agent receives an applicant profile via reset()
    2. Agent submits an underwriting decision via step()
    3. Environment grades the decision and returns reward + feedback

    Episodes are single-step: one applicant profile → one decision → done.
    """

    def __init__(self):
        """Initialize the environment with empty state."""
        self._current_task: Optional[TaskDefinition] = None
        self._observation: Optional[Observation] = None
        self._action_taken: Optional[Action] = None
        self._grading_result: Optional[GradingResult] = None
        self._done: bool = True
        self._episode_reward: float = 0.0

    def reset(
        self,
        task_id: Optional[str] = None,
        custom_profile: Optional[ApplicantProfile] = None
    ) -> State:
        """
        Reset the environment and load a new applicant profile.

        Args:
            task_id: ID of the task to load. If None, loads the first task.
            custom_profile: Optional ApplicantProfile to use instead of a predefined task.

        Returns:
            Initial State containing the observation (applicant profile + task description)
        """
        if custom_profile:
            # Handle custom profile
            self._current_task = TaskDefinition(
                task_id="custom_user_profile",
                name="Custom Applicant Profile",
                difficulty=TaskDifficulty.MEDIUM,
                description="Evaluate the manually entered applicant profile.",
                profile=custom_profile,
                ground_truth=generate_heuristic_ground_truth(custom_profile)
            )
        else:
            # Default to the first task if none specified
            if task_id is None:
                task_id = TASK_ORDER[0]

            # Load the task definition
            self._current_task = get_task(task_id)

        # Create the observation from the task's applicant profile
        self._observation = Observation.from_profile(
            profile=self._current_task.profile,
            task_description=self._current_task.description,
            task_id=self._current_task.task_id,
            task_difficulty=self._current_task.difficulty.value,
        )

        # Reset episode state
        self._action_taken = None
        self._grading_result = None
        self._done = False
        self._episode_reward = 0.0

        return self.state()

    def step(self, action: Action) -> tuple[State, float, bool, dict]:
        """
        Process the agent's underwriting decision and return results.

        This environment is single-step: the agent makes one decision per episode.
        After step() is called, the episode is done.

        Args:
            action: The agent's Action containing risk level, loan decision,
                    and interest rate tier.

        Returns:
            Tuple of (state, reward, done, info):
            - state: Updated State with grading results
            - reward: Float in [0.0, 1.0] with partial credit
            - done: Always True (single-step environment)
            - info: Dictionary with detailed grading breakdown

        Raises:
            RuntimeError: If called before reset() or after episode is done
        """
        # Validate environment state
        if self._current_task is None:
            raise RuntimeError(
                "Environment has not been reset. Call reset(task_id) first."
            )
        if self._done:
            raise RuntimeError(
                "Episode is already done. Call reset(task_id) to start a new episode."
            )

        # Store the action
        self._action_taken = action

        # Compute reward and get grading results
        reward, grading_result = compute_reward(
            action=action,
            ground_truth=self._current_task.ground_truth,
        )

        # Update episode state
        self._grading_result = grading_result
        self._episode_reward = reward
        self._done = True

        # Build info dictionary with detailed feedback
        info = {
            "task_id": self._current_task.task_id,
            "task_name": self._current_task.name,
            "task_difficulty": self._current_task.difficulty.value,
            "grading": {
                "risk_level_score": grading_result.risk_level_score,
                "loan_decision_score": grading_result.loan_decision_score,
                "interest_rate_score": grading_result.interest_rate_score,
                "consistency_bonus": grading_result.consistency_bonus,
                "total_score": grading_result.total_score,
            },
            "feedback": grading_result.feedback,
            "reward_breakdown": format_reward_breakdown(grading_result),
            "ground_truth": {
                "risk_level": self._current_task.ground_truth.risk_level.value,
                "loan_decision": self._current_task.ground_truth.loan_decision.value,
                "interest_rate_tier": self._current_task.ground_truth.interest_rate_tier.value,
                "explanation": self._current_task.ground_truth.explanation,
            },
        }

        return self.state(), reward, True, info

    def state(self) -> State:
        """
        Return the current environment state.

        This can be called at any time to inspect the environment's current state.

        Returns:
            State object containing observation, action taken, grading result, etc.
        """
        return State(
            observation=self._observation,
            action_taken=self._action_taken,
            grading_result=self._grading_result,
            done=self._done,
            current_task_id=(
                self._current_task.task_id if self._current_task else None
            ),
            episode_reward=self._episode_reward,
        )

    def get_available_tasks(self) -> list[dict]:
        """
        Return metadata about all available tasks.

        Returns:
            List of dictionaries with task metadata (id, name, difficulty, description)
        """
        tasks = get_all_tasks()
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty.value,
                "description": t.description,
            }
            for t in tasks
        ]

    @property
    def is_done(self) -> bool:
        """Whether the current episode has ended."""
        return self._done

    @property
    def current_task_id(self) -> Optional[str]:
        """ID of the current task, or None if not reset."""
        return self._current_task.task_id if self._current_task else None
