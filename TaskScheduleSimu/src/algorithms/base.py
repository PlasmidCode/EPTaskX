from abc import ABC, abstractmethod


class BaseScheduler(ABC):
    """Base class for scheduling algorithms."""

    def __init__(self, problem):
        self.problem = problem

    @abstractmethod
    def solve(self):
        """Return a scheduling result dictionary."""
        pass
