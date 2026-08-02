from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TrajectoryBatch:
    """States sampled on one shared time grid."""
    ## states uses [batch, time, state]; times uses [time].

    times: torch.Tensor
    states: torch.Tensor

    def __post_init__(self) -> None:
        if self.times.ndim != 1 or self.times.numel() < 2:
            raise ValueError("times must be one-dimensional with at least two values")
        if self.states.ndim != 3:
            raise ValueError("states must have shape [batch, time, state]")
        if self.states.shape[1] != self.times.numel():
            raise ValueError("the state time axis must match times")
        if self.states.shape[0] == 0 or self.states.shape[2] == 0:
            raise ValueError("batch and state dimensions cannot be empty")
        if not torch.is_floating_point(self.times) or not torch.is_floating_point(self.states):
            raise TypeError("times and states must be floating-point tensors")
        if self.times.dtype != self.states.dtype or self.times.device != self.states.device:
            raise ValueError("times and states must share a dtype and device")
        if not bool((self.times[1:] > self.times[:-1]).all()):
            raise ValueError("times must be strictly increasing")
        if not bool(torch.isfinite(self.times).all()) or not bool(torch.isfinite(self.states).all()):
            raise ValueError("times and states must be finite")

    @property
    def initial_state(self) -> torch.Tensor:
        """Return the initial state with shape [batch, state]."""

        return self.states[:, 0]

    def to(self, device: torch.device | str) -> "TrajectoryBatch":
        """Move the batch tensors to a device."""

        return TrajectoryBatch(self.times.to(device), self.states.to(device))
