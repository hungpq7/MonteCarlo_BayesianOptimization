import pandas as pd
import numpy as np
import torch

from torch import Tensor
from botorch.test_functions.base import BaseTestProblem

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# ============================================================
# Global config & HIGGS loading
# ============================================================
SEED = 0
rng = np.random.RandomState(SEED)

def load_higgs(n_samples: int = 10_000, seed: int = 0):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

    df = pd.read_csv(
        url,
        compression="gzip",
        header=None,
        nrows=n_samples,  # <-- crucial: do NOT read the full 11M rows
    )

    y = df.iloc[:, 0].values.astype(np.int32)
    X = df.iloc[:, 1:].values.astype(np.float32)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))

    return X[idx], y[idx]


# Load ~10k HIGGS samples and split once, reused by all problems
X_higgs, y_higgs = load_higgs(n_samples=10_000, seed=SEED)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_higgs,
    y_higgs,
    test_size=0.2,
    random_state=SEED,
    stratify=y_higgs,
)

# ============================================================
# Helper: apply sklearn models to a batch of points
# ============================================================
def _batch_apply(
    X: Tensor,
    dim: int,
    eval_one_fn,
) -> Tensor:
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
# 1) HiggsGBM: Gradient Boosting on HIGGS
#    dim = 3:
#      x[0] = n_estimators   in [50, 500]      (rounded to int)
#      x[1] = max_depth      in [1,  8]        (rounded to int)
#      x[2] = log10(learning_rate) in [-3, 0]  -> [1e-3, 1]
#    Objective: ROC-AUC on test set (maximize)
# ============================================================
class HiggsGBM(BaseTestProblem):
    dim = 3
    _bounds = [
        (50.0, 500.0),   # n_estimators
        (1.0, 8.0),      # max_depth
        (-3.0, 0.0),     # log10(learning_rate)
    ]
    continuous_inds = [0, 1, 2]
    discrete_inds = []
    categorical_inds = []
    _is_minimization_by_default = False  # maximize AUROC

    def __init__(
        self,
        noise_std: float | None = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        def eval_one(row: np.ndarray) -> float:
            n_estimators, max_depth, log10_lr = row.tolist()

            n_estimators = int(np.clip(round(n_estimators), 50, 500))
            max_depth = int(np.clip(round(max_depth), 1, 8))
            learning_rate = float(10.0 ** log10_lr)

            clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=SEED,
            )
            clf.fit(X_train_h, y_train_h)
            y_prob = clf.predict_proba(X_test_h)[:, 1]
            return roc_auc_score(y_test_h, y_prob)

        return _batch_apply(X, dim=self.dim, eval_one_fn=eval_one)

# ============================================================
# 2) HiggsDecisionTree: Decision Tree on HIGGS
#    dim = 3:
#      x[0] = max_depth         in [2, 40]   (rounded to int)
#      x[1] = min_samples_split in [2, 50]   (rounded to int)
#      x[2] = min_samples_leaf  in [1, 20]   (rounded to int)
#    Objective: ROC-AUC on test set (maximize)
# ============================================================
class HiggsDecisionTree(BaseTestProblem):
    dim = 3
    _bounds = [
        (2.0, 40.0),   # max_depth
        (2.0, 50.0),   # min_samples_split
        (1.0, 20.0),   # min_samples_leaf
    ]
    continuous_inds = [0, 1, 2]
    discrete_inds = []
    categorical_inds = []
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
            max_depth, min_samples_split, min_samples_leaf = row.tolist()

            max_depth = int(np.clip(round(max_depth), 2, 40))
            min_samples_split = int(np.clip(round(min_samples_split), 2, 50))
            min_samples_leaf = int(np.clip(round(min_samples_leaf), 1, 20))

            clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=SEED,
            )
            clf.fit(X_train_h, y_train_h)
            y_prob = clf.predict_proba(X_test_h)[:, 1]
            return roc_auc_score(y_test_h, y_prob)

        return _batch_apply(X, dim=self.dim, eval_one_fn=eval_one)

# ============================================================
# 3) HiggsRandomForest: Random Forest on HIGGS
#    dim = 3:
#      x[0] = n_estimators  in [50, 500]     (rounded to int)
#      x[1] = max_depth     in [2, 40]       (rounded to int)
#      x[2] = max_features  in [0.2, 1.0]    (fraction of features)
#    Objective: ROC-AUC on test set (maximize)
# ============================================================
class HiggsRandomForest(BaseTestProblem):
    dim = 3
    _bounds = [
        (50.0, 500.0),  # n_estimators
        (2.0, 40.0),    # max_depth
        (0.2, 1.0),     # max_features fraction
    ]
    continuous_inds = [0, 1, 2]
    discrete_inds = []
    categorical_inds = []
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
            n_estimators, max_depth, max_features = row.tolist()

            n_estimators = int(np.clip(round(n_estimators), 50, 500))
            max_depth = int(np.clip(round(max_depth), 2, 40))
            max_features = float(np.clip(max_features, 0.2, 1.0))

            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                random_state=SEED,
                n_jobs=-1,
            )
            clf.fit(X_train_h, y_train_h)
            y_prob = clf.predict_proba(X_test_h)[:, 1]
            return roc_auc_score(y_test_h, y_prob)

        return _batch_apply(X, dim=self.dim, eval_one_fn=eval_one)

# ============================================================
# 4) HiggsElasticNet: Logistic Regression (Elastic-Net penalty) on HIGGS
#    dim = 2:
#      x[0] = log10(C)     in [-4, 2]  -> C in [1e-4, 1e2]
#      x[1] = l1_ratio     in [0.0, 1.0]
#    Objective: ROC-AUC on test set (maximize)
# ============================================================
class HiggsElasticNet(BaseTestProblem):
    dim = 2
    _bounds = [
        (-4.0, 2.0),  # log10(C)
        (0.0, 1.0),   # l1_ratio
    ]
    continuous_inds = [0, 1]
    discrete_inds = []
    categorical_inds = []
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
            log10_C, l1_ratio = row.tolist()

            C = float(10.0 ** log10_C)
            l1_ratio = float(np.clip(l1_ratio, 0.0, 1.0))

            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=C,
                    l1_ratio=l1_ratio,
                    max_iter=2000,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            )
            clf.fit(X_train_h, y_train_h)
            y_prob = clf.predict_proba(X_test_h)[:, 1]
            return roc_auc_score(y_test_h, y_prob)

        return _batch_apply(X, dim=self.dim, eval_one_fn=eval_one)

# ============================================================
# 5) HiggsSVM: RBF SVM on HIGGS
#    dim = 2:
#      x[0] = log10(C)     in [-2, 3]   -> C in [1e-2, 1e3]
#      x[1] = log10(gamma) in [-4, 1]   -> gamma in [1e-4, 1e1]
#    Objective: ROC-AUC on test set (maximize)
# ============================================================
class HiggsSVM(BaseTestProblem):
    dim = 2
    _bounds = [
        (-2.0, 3.0),  # log10(C)
        (-4.0, 1.0),  # log10(gamma)
    ]
    continuous_inds = [0, 1]
    discrete_inds = []
    categorical_inds = []
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
                    probability=True,
                    random_state=SEED,
                ),
            )
            clf.fit(X_train_h, y_train_h)
            y_prob = clf.predict_proba(X_test_h)[:, 1]
            return roc_auc_score(y_test_h, y_prob)

        return _batch_apply(X, dim=self.dim, eval_one_fn=eval_one)

# ============================================================
# 6) HiggsMLP: MLPClassifier on HIGGS
#    dim = 3:
#      x[0] = hidden_layer_size     in [50, 400]          (rounded to int)
#      x[1] = log10(alpha)         in [-6, -1]           -> [1e-6, 1e-1]
#      x[2] = log10(learning_rate) in [-4, -1]           -> [1e-4, 1e-1]
#    Objective: ROC-AUC on test set (maximize)
# ============================================================
class HiggsMLP(BaseTestProblem):
    dim = 3
    _bounds = [
        (50.0, 400.0),   # hidden_layer_size
        (-6.0, -1.0),    # log10(alpha)
        (-4.0, -1.0),    # log10(learning_rate_init)
    ]
    continuous_inds = [0, 1, 2]
    discrete_inds = []
    categorical_inds = []
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
            hidden_layer_size, log10_alpha, log10_lr = row.tolist()

            hidden_layer_size = int(np.clip(round(hidden_layer_size), 50, 400))
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
            clf.fit(X_train_h, y_train_h)
            y_prob = clf.predict_proba(X_test_h)[:, 1]
            return roc_auc_score(y_test_h, y_prob)

        return _batch_apply(X, dim=self.dim, eval_one_fn=eval_one)

