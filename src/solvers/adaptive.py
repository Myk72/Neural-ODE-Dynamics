from collections.abc import Callable

import torch
from torchdiffeq import odeint

from solvers.fixed_step import _evaluate, _validate_inputs


def solve_adaptive(
    dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    times: torch.Tensor,
    *,
    method: str = "dopri5",
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> torch.Tensor:
    """Integrate at requested times with an adaptive differentiable solver."""

    _validate_inputs(initial_state, times)
    if rtol <= 0 or atol <= 0:
        raise ValueError("rtol and atol must be positive")

    def checked_dynamics(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return _evaluate(dynamics, time, state)

    solution = odeint(
        checked_dynamics,
        initial_state,
        times,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if solution.shape != (times.numel(), *initial_state.shape):
        raise RuntimeError("the adaptive solver returned an unexpected shape")
    return solution
