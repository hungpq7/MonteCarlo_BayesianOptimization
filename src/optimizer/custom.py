from botorch.optim import optimize_acqf
import torch
torch.set_default_dtype(torch.double)

from .base import BaseOptimizer
from ..acquisition.mc_ei import (
    ExpectedImprovementTwoStepLookahead,
    qExpectedImprovementTwoStepLookahead,
)

from ..acquisition.mlmc_ei import (
    qEIMLMCTwoStep,
    qEIMLMCThreeStep,
    optimize_mlmc,
    optimize_mlmc_three,
)

class EITwoStepMCOptimizer(BaseOptimizer):
    def gen_new_candidate(self, model, best_f, **kwargs):
        acqf = ExpectedImprovementTwoStepLookahead(model=model)

        x_new, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
        )

        y_new = self.blackbox(x_new).unsqueeze(-1)
        return x_new, y_new

class qEITwoStepMCOptimizer(BaseOptimizer):
    def gen_new_candidate(self, model, best_f, **kwargs):
        acqf = ExpectedImprovementTwoStepLookahead(model=model, batch_sizes=[2])

        candidates = kwargs.get('candidates', 5)
        x_new, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=candidates,
            num_restarts=10,
            raw_samples=128,
            return_best_only=True,
        )

        y_new = self.blackbox(x_new).unsqueeze(-1)
        return x_new, y_new

class qEITwoStepMLMCOptimizer(BaseOptimizer):
    def gen_new_candidate(self, model, best_f, **kwargs):
        acqf = qEIMLMCTwoStep(
            model=model,
            bounds=self.bounds,
            num_restarts=30,
            raw_samples=100,
            q=1,
            batch_sizes=[2]
        )

        x_new, _, _ = optimize_mlmc(
            inc_function=acqf,
        )

        y_new = self.blackbox(x_new).unsqueeze(-1)
        return x_new, y_new

class qEIThreeStepMLMCOptimizer(BaseOptimizer):
    def gen_new_candidate(self, model, best_f, **kwargs):
        acqf = qEIMLMCThreeStep(
            model=model,
            bounds=self.bounds,
            num_restarts=10,
            raw_samples=256,
            q=1,
            batch_sizes=[1, 1]
        )

        x_new, _, _ = optimize_mlmc_three(
            inc_function=acqf,
        )

        y_new = self.blackbox(x_new).unsqueeze(-1)
        return x_new, y_new