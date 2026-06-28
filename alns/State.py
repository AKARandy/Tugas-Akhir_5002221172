class State:
    """Minimal ALNS-compatible base class for local state implementations."""

    def objective(self) -> float:
        raise NotImplementedError
