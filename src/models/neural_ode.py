import torch
from torch import nn

from solvers import solve_fixed


class NeuralVectorField(nn.Module):
    """MLP that maps the current state to its instantaneous derivative."""

    def __init__(self, state_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        if state_dim < 1 or hidden_dim < 1:
            raise ValueError("state_dim and hidden_dim must be positive")

        self.state_dim = state_dim

        # simple mlp 
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        
        self.evaluation_count = 0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear_layers = [layer for layer in self.network if isinstance(layer, nn.Linear)]
        for layer in linear_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Begin near the persistence baseline so the first rollout is stable
        nn.init.uniform_(linear_layers[-1].weight, -1e-3, 1e-3)

    def forward(self, _time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"expected state dimension {self.state_dim}, received {state.shape[-1]}")
        
        self.evaluation_count += 1
        return self.network(state)

    def reset_evaluation_counter(self) -> None:
        self.evaluation_count = 0


class NeuralODE(nn.Module):
    def __init__(self, vector_field: NeuralVectorField, *, method: str = "rk4") -> None:
        super().__init__()
        if method not in {"euler", "rk4"}:
            raise ValueError("NeuralODE method must be 'euler' or 'rk4'")
        self.vector_field = vector_field
        self.method = method

    def forward(self, initial_state: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if initial_state.ndim != 2:
            raise ValueError("initial_state must have shape [batch, state]")

        solved_time_states = solve_fixed(self.vector_field, initial_state, times, method=self.method)
        return solved_time_states.movedim(0, 1)
