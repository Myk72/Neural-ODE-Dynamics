from collections.abc import Callable

import torch


def _validate_inputs(initial_state: torch.Tensor, times: torch.Tensor) -> None:
    if not isinstance(initial_state, torch.Tensor) or not isinstance(times, torch.Tensor):
        raise TypeError("initial_state and times must be torch tensors")
    if times.ndim != 1 or times.numel() < 2:
        raise ValueError("times must be one-dimensional with at least two values")
    if initial_state.numel() == 0:
        raise ValueError("initial_state cannot be empty")
    if not torch.is_floating_point(initial_state) or not torch.is_floating_point(times):
        raise TypeError("initial_state and times must be floating-point tensors")
    if initial_state.dtype != times.dtype or initial_state.device != times.device:
        raise ValueError("initial_state and times must share a dtype and device")
    if not bool(torch.isfinite(initial_state).all()) or not bool(torch.isfinite(times).all()):
        raise ValueError("initial_state and times must be finite")
    if not bool((times[1:] > times[:-1]).all()):
        raise ValueError("times must be strictly increasing")


def _evaluate(
    dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    time: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    derivative = dynamics(time, state)
    if not isinstance(derivative, torch.Tensor):
        raise TypeError("dynamics must return a torch tensor")
    if derivative.shape != state.shape:
        raise ValueError(
            f"dynamics returned shape {tuple(derivative.shape)} "
            f"for state shape {tuple(state.shape)}"
        )
    if derivative.dtype != state.dtype or derivative.device != state.device:
        raise ValueError("dynamics must preserve the state dtype and device")
    if not bool(torch.isfinite(derivative).all()):
        raise ValueError("dynamics returned non-finite values")
    return derivative


def _euler_step(
    dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    time: torch.Tensor,
    state: torch.Tensor,
    step_size: torch.Tensor,
) -> torch.Tensor:
    return state + step_size * _evaluate(dynamics, time, state)


def _rk4_step(
    dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    time: torch.Tensor,
    state: torch.Tensor,
    step_size: torch.Tensor,
) -> torch.Tensor:
    half_step = step_size / 2
    k1 = _evaluate(dynamics, time, state)
    k2 = _evaluate(dynamics, time + half_step, state + half_step * k1)
    k3 = _evaluate(dynamics, time + half_step, state + half_step * k2)
    k4 = _evaluate(dynamics, time + step_size, state + step_size * k3)
    return state + step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6


_solver_methods = {
    "euler": _euler_step,
    "rk4": _rk4_step,
}


def solve_fixed(
    dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    times: torch.Tensor,
    *,
    method: str = "rk4",
) -> torch.Tensor:

    _validate_inputs(initial_state, times)
    if method not in _solver_methods:
        raise ValueError(f"unknown method {method!r}; choose from {tuple(_solver_methods)}")

    step = _solver_methods[method]
    states = [initial_state]
    current = initial_state
    for start, end in zip(times[:-1], times[1:]):
        current = step(dynamics, start, current, end - start)
        states.append(current)

    return torch.stack(states)
