import logging
from math import isclose

logger = logging.getLogger(__name__)


class EarlyStopper:
    """Early stopping implementation that tracks validation metric progress.

    It tracks the best value of a validation metric and signals when to stop
    training if no significant improvement is observed for a specified number
    of consecutive evaluations.

    Attributes:
        best (float): Best observed metric value so far
        counter (int): Number of consecutive evaluations without improvement
        improved (bool): Whether the metric improved in the last step
        min_delta (float): Minimum **relative** improvement required
            to be considered significant (expressed as fraction)
        patience (int): Number of consecutive non-improving evaluations that
            can elapse before stopping
        stop (bool): Signal to stop training (evaluations without improvement
            exceed patience)
        save (bool): Signal that model improved & should be saved

    Examples:
        >>> stopper = EarlyStopper(patience=5, min_delta=0.01)
        >>> for epoch in range(100):
        >>>     train_model()
        >>>     val_loss = validate_model()
        >>>     stopper.step(val_loss)
        >>>     if stopper.save:
        >>>         save_model()
        >>>     if stopper.stop:
        >>>         print(f"Early stopping at epoch {epoch}")
        >>>         break
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.01):
        self.best = float("inf")
        self.counter = 0
        self.improved = False
        self.min_delta = min_delta
        self.patience = patience

    def step(self, last: float) -> None:
        is_improvement = last < self.best
        is_significant = not isclose(last, self.best, rel_tol=self.min_delta)

        if is_improvement and is_significant:
            self.best = last
            self.counter = 0
            self.improved = True
        else:
            self.counter += 1
            self.improved = False

    @property
    def stop(self) -> bool:
        """True if `patience` evaluations elapsed without improvement."""
        return self.counter >= self.patience

    @property
    def save(self) -> bool:
        "True if the metric significantly improved in the latest step."
        return self.improved
