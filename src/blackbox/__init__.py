import numpy as np
import pandas as pd
import torch

from torch import Tensor
from botorch.test_functions.base import BaseTestProblem
from botorch.test_functions.synthetic import SyntheticTestFunction

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# ============================================================
# Global config & data loading (done once, outside classes)
# ============================================================
SEED = 0
rng = np.random.RandomState(SEED)

# 1) Breast Cancer (classification) -------------------------------------------
X_bc, y_bc = load_breast_cancer(return_X_y=True)
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_bc,
    y_bc,
    test_size=0.2,
    random_state=SEED,
    stratify=y_bc,
)

# 2) Wine Quality (Red) (regression) ------------------------------------------
url_red = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
wine_red_df = pd.read_csv(url_red, sep=";")
X_wq = wine_red_df.drop(columns=["quality"]).values
y_wq = wine_red_df["quality"].values

X_train_wq, X_test_wq, y_train_wq, y_test_wq = train_test_split(
    X_wq,
    y_wq,
    test_size=0.2,
    random_state=SEED,
)

# 3) Wine (classification) ----------------------------------------------------
X_wine, y_wine = load_wine(return_X_y=True)
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine,
    y_wine,
    test_size=0.2,
    random_state=SEED,
    stratify=y_wine,
)


# ============================================================
# Base helper: apply sklearn models to a batch of points
# ============================================================
def _batch_apply(
    X: Tensor,
    dim: int,
    eval_one_fn,
) -> Tensor:
    """
    Helper to map a batch of hyperparameters to scalar scores.

    X: (batch_shape) x dim tensor
    eval_one_fn: function taking a 1D numpy array of length dim -> float score

    Returns:
        Tensor with shape batch_shape (same device & dtype as X).
    """
    # Ensure double, handle arbitrary batch shape
    X = X.to(dtype=torch.double)
    device = X.device
    orig_shape = X.shape[:-1]  # batch_shape
    X_flat = X.reshape(-1, dim)  # (N, dim)

    X_np = X_flat.detach().cpu().numpy()
    scores = []
    for row in X_np:
        score = eval_one_fn(row)
        scores.append(float(score))

    scores_tensor = torch.tensor(scores, dtype=X.dtype, device=device)
    return scores_tensor.reshape(orig_shape)


# ============================================================
# 1) BreastCancerSVM: Breast Cancer + RBF SVM
#    dim = 2:
#      x[0] = log10(C)     in [-2, 3]   -> C in [1e-2, 1e3]
#      x[1] = log10(gamma) in [-4, 1]   -> gamma in [1e-4, 1e1]
#    Objective: accuracy on test set (maximize)
# ============================================================
class BreastCancerSVM(BaseTestProblem):
    
    dim = 2
    # BoTorch expects _bounds as list[(low, high)] and will build a 2 x d tensor
    _bounds = [(-2.0, 3.0), (-4.0, 1.0)]
    # All dimensions are treated as continuous; we round internally if needed
    continuous_inds = [0, 1]
    discrete_inds = []
    categorical_inds = []
    # This is a maximization problem (accuracy)
    _is_minimization_by_default = False

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        def eval_one(row: np.ndarray) -> float:
            log10_C, log10_gamma = row.tolist()
            C = float(10.0 ** log10_C)
            gamma = float(10.0 ** log10_gamma)

            clf = make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    random_state=SEED,
                ),
            )
            clf.fit(X_train_bc, y_train_bc)
            y_prob = clf.predict_proba(X_test_bc)[:, 1]
            return clf.score(y_test_bc, y_prob)  # auroc

        return _batch_apply(X, dim=self.dim, eval_one_fn=eval_one)


# ============================================================
# 2) WineQualityRF: Wine Quality (Red) + RandomForestRegressor
#    dim = 3:
#      x[0] = n_estimators      in [10, 300]     (rounded to int)
#      x[1] = max_depth         in [2, 20]       (rounded to int)
#      x[2] = min_samples_split in [2, 20]       (rounded to int)
#    Objective: R^2 on test set (maximize)
# ============================================================
class WineQualityRF(BaseTestProblem):
    dim = 3
    _bounds = [
        (10.0, 300.0),  # n_estimators
        (2.0, 20.0),    # max_depth
        (2.0, 20.0),    # min_samples_split
    ]
    continuous_inds = [0, 1, 2]  # treat as continuous; discretize inside
    discrete_inds = []
    categorical_inds = []
    _is_minimization_by_default = False  # maximize R^2

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        def eval_one(row: np.ndarray) -> float:
            n_estimators, max_depth, min_samples_split = row.tolist()

            n_estimators = int(np.clip(round(n_estimators), 10, 300))
            max_depth = int(np.clip(round(max_depth), 2, 20))
            min_samples_split = int(np.clip(round(min_samples_split), 2, 20))

            reg = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=SEED,
                n_jobs=-1,
            )
            reg.fit(X_train_wq, y_train_wq)
            return reg.score(X_test_wq, y_test_wq)  # R^2

        return _batch_apply(X, dim=self.dim, eval_one_fn=eval_one)


# ============================================================
# 3) WineMLP: Wine (classification) + MLPClassifier
#    dim = 3:
#      x[0] = hidden_layer_size        in [5, 200]           (rounded to int)
#      x[1] = log10(alpha)            in [-6, -1]           -> [1e-6, 1e-1]
#      x[2] = log10(learning_rate)    in [-4, -1]           -> [1e-4, 1e-1]
#    Objective: accuracy on test set (maximize)
# ============================================================
class WineMLP(BaseTestProblem):
    dim = 3
    _bounds = [
        (5.0, 200.0),   # hidden_layer_size
        (-6.0, -1.0),   # log10(alpha)
        (-4.0, -1.0),   # log10(learning_rate_init)
    ]
    continuous_inds = [0, 1, 2]
    discrete_inds = []
    categorical_inds = []
    _is_minimization_by_default = False  # maximize accuracy

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        def eval_one(row: np.ndarray) -> float:
            hidden_layer_size, log10_alpha, log10_lr = row.tolist()

            hidden_layer_size = int(np.clip(round(hidden_layer_size), 5, 200))
            alpha = float(10.0 ** log10_alpha)
            learning_rate_init = float(10.0 ** log10_lr)

            clf = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(hidden_layer_size,),
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    max_iter=1000,
                    random_state=SEED,
                    early_stopping=True,
                ),
            )
            clf.fit(X_train_wine, y_train_wine)
            return clf.score(X_test_wine, y_test_wine)  # accuracy

        return _batch_apply(X, dim=self.dim, eval_one_fn=eval_one)


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    # Breast cancer + SVM
    bc = BreastCancerSVM()
    print("BreastCancerSVM bounds:", bc.bounds)

    # Single point (dim=2) -> scalar
    x0 = torch.tensor([-1.0, -3.0], dtype=torch.double)  # log10(C), log10(gamma)
    print("bc(x0) =", bc(x0))

    # Batch of points (N, d) -> shape (N,)
    X_batch = torch.stack([
        torch.tensor([-1.0, -3.0], dtype=torch.double),
        torch.tensor([1.0, -2.0], dtype=torch.double),
    ], dim=0)
    print("bc(X_batch) =", bc(X_batch))

    # WineMLP
    mlp = WineMLP()
    print("WineMLP bounds:", mlp.bounds)
