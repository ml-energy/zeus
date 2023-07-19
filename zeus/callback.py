# Copyright (C) 2023 Jae-Won Chung <jwnchung@umich.edu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
