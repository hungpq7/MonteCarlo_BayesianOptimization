from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
torch.set_default_dtype(torch.double)

from botorch import settings
from botorch.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    qKnowledgeGradient,
    MCAcquisitionObjective,
    qExpectedImprovement,
    qMultiStepLookahead
)
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.multi_step_lookahead import make_best_f
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform, concatenate_pending_points
from torch import Tensor
from torch.nn import ModuleList

TAcqfArgConstructor = Callable[[Model, Tensor], Dict[str, Any]]

eps = 0.25
N = np.round((1 / np.power(eps, 2))).astype(int)
M = N

sampler = IIDNormalSampler(sample_shape=torch.Size([N]), resample=False)
inner_sampler = IIDNormalSampler(sample_shape=torch.Size([M]), resample=False)

class ExpectedImprovementTwoStepLookahead(qKnowledgeGradient):
    r"""two-step lookahead expected improvement
    implemented in a one-shot fashion"""

    def __int__(
            self,
            model: Model,
            num_fantasies: Optional[int] = None,
            sampler: Optional[MCSampler] = sampler,
            objective: Optional[MCAcquisitionObjective] = None,
            inner_sampler: Optional[MCSampler] = inner_sampler,
            X_pending: Optional[Tensor] = None,
            current_value: Optional[Tensor] = None
    ) -> None:
        super().__init__(
            model,
            num_fantasies,
            sampler,
            objective,
            inner_sampler,
            X_pending,
            current_value
        )

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate one step-lookahead qEI on the candidate set 'X'

        Args:
            X: A `b x (q + num_fantasies) x d` Tensor with `b` t-batches of
                `q + num_fantasies` design points each. We split this X tensor
                into two parts in the `q` dimension (`dim=-2`). The first `q`
                are the q-batch of design points and the last num_fantasies are
                the current solutions of the inner optimization problem.

                `X_fantasies = X[..., -num_fantasies:, :]`
                `X_fantasies.shape = b x num_fantasies x d`

                `X_actual = X[..., :-num_fantasies, :]`
                `X_actual.shape = b x q x d`

        Returns:
            A Tensor of shape `b`.
            For t-batch b, the one-step lookahead EI value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
        """
        X_actual, X_fantasies = _split_fantasy_points(
            X=X, n_f=self.num_fantasies
        )

        current_value = (
            self.current_value if self.current_value else self.model.train_targets.max()
        )

        ei = ExpectedImprovement(model=self.model, best_f=current_value)
        zero_step_ei = ei(X_actual)

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_actual = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )

        # construct the fantasy model of shape `num_fantasies x b`
        fantasy_model = self.model.fantasize(
            X=X_actual, sampler=self.sampler, observation_noise=False
        )

        best_f = fantasy_model.train_targets.max(dim=-1)[0]

        if not self.inner_sampler:
            one_step_ei = ExpectedImprovement(model=fantasy_model, best_f=best_f)
        else:
            one_step_ei = qExpectedImprovement(model=fantasy_model,
                                               sampler=self.inner_sampler,
                                               best_f=best_f)

        with settings.propagate_grads(True):
            values = one_step_ei(X=X_fantasies)

        one_step_ei_avg = values.mean(dim=0)

        return zero_step_ei + one_step_ei_avg