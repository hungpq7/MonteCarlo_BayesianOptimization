from .base import BaseOptimizer

import torch
torch.set_default_dtype(torch.double)

from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf

class EIOptimizer(BaseOptimizer):
    def gen_new_candidate(self, model, best_f, **kwargs):
        acqf = ExpectedImprovement(model=model, best_f=best_f, maximize=True)

        x_new, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
        )

        y_new = self.blackbox(x_new).unsqueeze(-1)
        return x_new, y_new

class qEIOptimizer(BaseOptimizer):
    def gen_new_candidate(self, model, best_f, **kwargs):
        acqf = qExpectedImprovement(model=model, best_f=best_f, maximize=True)

        candidates = kwargs.get('candidates', 5)
        x_new, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=candidates,
            num_restarts=30,
            raw_samples=100,
        )

        y_new = self.blackbox(x_new).unsqueeze(-1)
        return x_new, y_new

class qLogEIOptimizer(BaseOptimizer):
    def gen_new_candidate(self, model, best_f, **kwargs):
        acqf = qLogExpectedImprovement(model=model, best_f=best_f)

        candidates = kwargs.get('candidates', 5)
        x_new, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=candidates,
            num_restarts=10,
            raw_samples=128,
        )

        y_new = self.blackbox(x_new).unsqueeze(-1)
        return x_new, y_new
