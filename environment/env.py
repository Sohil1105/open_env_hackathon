"""
OpenEnv environment for Financial Loan Underwriting & Risk Assessment.

Implements reset / step / state following the OpenEnv spec.
A full-lifecycle episode (task_id=None) runs 8 sequential loan stages;
passing a specific task_id runs a single-stage episode.
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
    OpenEnv environment for loan underwriting.

    Full lifecycle (task_id=None): 8 stages, one step per stage, returns
    per-stage reward so the agent gets intermediate feedback throughout.
    Single task (task_id specified): 1-step episode for targeted evaluation.
    """

    def __init__(self):
        """Initialize the environment with empty state."""
        self._current_task: Optional[TaskDefinition] = None
        self._observation: Optional[Observation] = None
        self._action_taken: Optional[Action] = None
        self._grading_result: Optional[GradingResult] = None
        self._done: bool = True
        self._episode_reward: float = 0.0
        self._current_step: int = 0
        self._max_steps: int = 1
        self._task_sequence: list[str] = []
        self._step_rewards: list[float] = []

    def reset(
        self,
        task_id: Optional[str] = None,
        custom_profile: Optional[ApplicantProfile] = None
    ) -> State:
        """Reset and return the initial State. task_id=None starts the full 8-stage lifecycle."""
        if custom_profile:
            self._current_task = TaskDefinition(
                task_id="custom_user_profile",
                name="Custom Applicant Profile",
                difficulty=TaskDifficulty.MEDIUM,
                description="Evaluate the manually entered applicant profile.",
                profile=custom_profile,
                ground_truth=generate_heuristic_ground_truth(custom_profile)
            )
            self._task_sequence = ["custom_user_profile"]
        else:
            if task_id:
                self._current_task = get_task(task_id)
                self._task_sequence = [task_id]
            else:
                # Full 8-stage lifecycle
                self._task_sequence = TASK_ORDER
                self._current_task = get_task(self._task_sequence[0])

        self._current_step = 0
        self._max_steps = len(self._task_sequence)
        self._step_rewards = []
        self._action_taken = None
        self._grading_result = None
        self._done = False
        self._episode_reward = 0.0

        self._update_observation()

        return self.state()

    def _update_observation(self):
        """Update the observation based on the current task in the sequence."""
        if self._current_step < self._max_steps:
            task_id = self._task_sequence[self._current_step]
            self._current_task = get_task(task_id)
            self._observation = Observation.from_profile(
                profile=self._current_task.profile,
                task_description=self._current_task.description,
                task_id=self._current_task.task_id,
                task_difficulty=self._current_task.difficulty.value,
            )
        else:
            self._observation = None

        return self.state()

    def step(self, action: Action) -> tuple[State, float, bool, dict]:
        """
        Grade the action for the current stage and advance to the next.

        Returns (state, reward, done, info). In a lifecycle episode, done=False
        until all 8 stages are complete; info includes per-component scores for
        each stage so the agent receives meaningful intermediate feedback.
        """
        if self._current_task is None:
            raise RuntimeError(
                "Environment has not been reset. Call reset(task_id) first."
            )
        if self._done:
            raise RuntimeError(
                "Episode is already done. Call reset(task_id) to start a new episode."
            )

        self._action_taken = action

        step_reward, grading_result = compute_reward(
            action=action,
            ground_truth=self._current_task.ground_truth,
        )

        self._grading_result = grading_result
        self._step_rewards.append(step_reward)
        self._episode_reward = sum(self._step_rewards) / len(self._step_rewards)

        self._current_step += 1
        if self._current_step >= self._max_steps:
            self._done = True
        else:
            self._update_observation()

        info = {
            "task_id": self._current_task.task_id,
            "task_name": self._current_task.name,
            "task_difficulty": self._current_task.difficulty.value,
            "current_step": self._current_step,
            "total_steps": self._max_steps,
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

        return self.state(), step_reward, self._done, info

    def state(self) -> State:
        """Return the current environment state (safe to call at any time)."""
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
        """Return metadata for all available tasks."""
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
