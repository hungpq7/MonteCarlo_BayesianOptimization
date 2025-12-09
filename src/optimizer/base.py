from concurrent.futures import ThreadPoolExecutor
import os
import time

import numpy as np
import torch
torch.set_default_dtype(torch.double)
from torch import Tensor

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


class BaseOptimizer:
    def __init__(
        self,
        blackbox=None,
        n_init=10,
        max_iter=50,
    ):
        self.blackbox = blackbox
        self.dim = blackbox.dim
        self.bounds = blackbox.bounds.to(dtype=torch.double)
        self.n_init = n_init
        self.max_iter = max_iter

    def gen_initial_data(self, size, dim):
        x_train = torch.rand(size, dim, dtype=torch.double)
        y_train = self.blackbox(x_train).unsqueeze(-1)
        return x_train, y_train

    def build_model(self, x_train, y_train):
        model = SingleTaskGP(
            x_train, y_train,
            input_transform=Normalize(d=self.dim),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return model, mll

    def fit_model(self, x_train, y_train, state_dict=None):
        model, mll = self.build_model(x_train, y_train)
        if state_dict is not None:
            # Load previous GP's parameters for warm start
            model.load_state_dict(state_dict)
        
        # Fit the model with LBFGS-B
        fit_gpytorch_mll(mll)

        # Save updated GP's parameters to warm start next iteration
        new_state_dict = model.state_dict()
        return model, mll, new_state_dict
    
    def gen_new_candidate(self, model, best_f, **kwargs):
        x_new = torch.rand(1, self.dim, dtype=torch.double)
        y_new = self.blackbox(x_new).unsqueeze(-1)
        return x_new, y_new
    
    def run_bo(self, print_every=0, **kwargs):
        start = time.time()

        x_train, y_train = self.gen_initial_data(self.n_init, self.dim)
        model, mll, state_dict = self.fit_model(x_train, y_train)

        batch_size = kwargs.get('batch_size', 1)
        n_iter = (self.max_iter - self.n_init) // batch_size

        for i in range(n_iter):
            best_f = y_train.max().item()
            x_new, y_new = self.gen_new_candidate(model, best_f, **kwargs)

            x_train = torch.cat([x_train, x_new], dim=0)
            y_train = torch.cat([y_train, y_new], dim=0)

            # Rebuild + warm-start + refit on all data
            model, mll, state_dict = self.fit_model(x_train, y_train, state_dict=state_dict)

            if print_every > 0:
                if (i+1) % print_every == 0:
                    y_best = y_train.max().item()
                    print(f"Iter {i+1} | Current best: {y_best:.4f}")

        y_best = y_train.max().item()
        duration = time.time() - start
        return y_best, duration


    def benchmark(self, runs=10, n_jobs=None, **kwargs):
        if runs <= 0:
            return np.array([]), np.array([])

        # Decide number of workers
        if n_jobs is None:
            n_jobs = min(runs, os.cpu_count())

        # Sequential fallback (original behaviour)
        if n_jobs <= 1:
            best_vals = []
            durations = []
            for _ in range(runs):
                best_val, duration = self.run_bo(**kwargs)
                best_vals.append(best_val)
                durations.append(duration)
            return np.array(best_vals), np.array(durations)

        # Parallel execution
        best_vals = np.empty(runs, dtype=float)
        durations = np.empty(runs, dtype=float)

        # Launch all runs in parallel
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(self.run_bo, **kwargs)
                for _ in range(runs)
            ]
            # Preserve order by index in `futures`
            for i, fut in enumerate(futures):
                best_val, duration = fut.result()
                best_vals[i] = best_val
                durations[i] = duration

        return best_vals, durations