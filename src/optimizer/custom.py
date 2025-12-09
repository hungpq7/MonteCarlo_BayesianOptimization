from botorch.optim import optimize_acqf
import torch
torch.set_default_dtype(torch.double)

from .base import BaseOptimizer
from ..acquisition.mcei import (
    ExpectedImprovementTwoStepLookahead,
    qExpectedImprovementTwoStepLookahead,
)

class EITwoStepMCOptimizer(BaseOptimizer):
    def gen_new_candidate(self, model, best_f, **kwargs):
        acqf = ExpectedImprovementTwoStepLookahead(model=model)

        batch_size = kwargs.get('batch_size', 5)
        x_new, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
        )

        y_new = self.blackbox(x_new).unsqueeze(-1)
        return x_new, y_new