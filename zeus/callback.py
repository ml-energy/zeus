"""Infrastructure for calling callbacks."""

from __future__ import annotations


class Callback:
    """Base class for callbacks."""

    def on_train_begin(self) -> None:
        """Called at the beginning of training."""

    def on_train_end(self) -> None:
        """Called at the end of training."""

    def on_epoch_begin(self) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(self) -> None:
        """Called at the end of each epoch."""

    def on_step_begin(self) -> None:
        """Called at the beginning of each step."""

    def on_step_end(self) -> None:
        """Called at the end of each step."""

    def on_evaluate(self, metric: float) -> None:
        """Called after evaluating the model."""

    def on_instruction_begin(self, name: str) -> None:
        """Called at the beginning of pipeline instructions like forward or backward."""

    def on_instruction_end(self, name: str) -> None:
        """Called at the end of pipeline instructions like forward or backward."""


class CallbackSet(Callback):
    """A set of callbacks."""

    def __init__(self, callbacks: list[Callback]) -> None:
        """Initialize the callback set."""
        self.callbacks = callbacks

    def on_train_begin(self) -> None:
        """Called at the beginning of training."""
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self) -> None:
        """Called at the end of training."""
        for callback in self.callbacks:
            callback.on_train_end()

    def on_epoch_begin(self) -> None:
        """Called at the beginning of each epoch."""
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self) -> None:
        """Called at the end of each epoch."""
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_step_begin(self) -> None:
        """Called at the beginning of each step."""
        for callback in self.callbacks:
            callback.on_step_begin()

    def on_step_end(self) -> None:
        """Called at the end of each step."""
        for callback in self.callbacks:
            callback.on_step_end()

    def on_evaluate(self, metric: float) -> None:
        """Called after evaluating the model."""
        for callback in self.callbacks:
            callback.on_evaluate(metric)

    def on_instruction_begin(self, name: str) -> None:
        """Called at the beginning of pipeline instructions like forward or backward."""
        for callback in self.callbacks:
            callback.on_instruction_begin(name)

    def on_instruction_end(self, name: str) -> None:
        """Called at the end of pipeline instructions like forward or backward."""
        for callback in self.callbacks:
            callback.on_instruction_end(name)
