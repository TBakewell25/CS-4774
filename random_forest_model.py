import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterSampler


class BusDelayRandomForest:

    def __init__(self, n_estimators: int = 100, max_depth: int = 20,
                 min_samples_leaf: int = 1, max_samples: float = 0.5,
                 random_state: int = 42):
        self.rf = RandomForestRegressor(
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            min_samples_leaf = min_samples_leaf,
            max_samples      = max_samples,
            n_jobs           = -1,
            random_state     = random_state,
        )
        self.feature_names: list[str] = []

    # Training 

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: list[str] | None = None) -> "BusDelayRandomForest":
        if feature_names:
            self.feature_names = feature_names
        self.rf.fit(X_train, y_train)
        return self

    # Evaluation 

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 split_name: str = "test") -> dict:
        preds = self.rf.predict(X)
        mae   = mean_absolute_error(y, preds)
        rmse  = mean_squared_error(y, preds) ** 0.5
        print(f"[RandomForest] {split_name} -> MAE: {mae:.2f}s   RMSE: {rmse:.2f}s")
        return {"mae": mae, "rmse": rmse}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict(X)

    # Hyperparameter tuning

    @classmethod
    def random_search(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        n_iter:  int = 20,
        max_samples: float = 0.5,
        random_state: int = 42,
    ) -> "BusDelayRandomForest":

        param_dist = {
            "n_estimators":    [50, 100, 200],
            "max_depth":       [10, 20, 30],
            "min_samples_leaf": [5, 20, 50],
        }

        best_mae   = float("inf")
        best_model = None
        best_params = {}
        rng = np.random.RandomState(random_state)

        sampler = list(ParameterSampler(param_dist, n_iter=n_iter,
                                        random_state=rng))
        print(f"Random-search over {n_iter} hyperparameter combinations ...")
        for i, params in enumerate(sampler, 1):
            model = cls(**params, max_samples=max_samples, random_state=random_state)
            model.fit(X_train, y_train)
            results = model.evaluate(X_val, y_val, split_name=f"val (iter {i})")
            if results["mae"] < best_mae:
                best_mae    = results["mae"]
                best_model  = model
                best_params = params
                print(f"   New best params: {params}  MAE={best_mae:.2f}s")

        print(f"\nBest hyperparams: {best_params}   Val MAE: {best_mae:.2f}s")
        return best_model

    # Feature importance 
    def feature_importances(self) -> dict:
        # Returns feature importance descending
        imp = self.rf.feature_importances_
        names = self.feature_names if self.feature_names else \
                [f"feat_{i}" for i in range(len(imp))]
        ranked = sorted(zip(names, imp), key=lambda x: -x[1])
        print("\nTop-10 feature importances:")
        for name, score in ranked[:10]:
            bar = "#" * int(score * 50)
            print(f"  {name:30s} {score:.4f}  {bar}")
        return dict(ranked)

    # Persistence

    def save(self, path: str = "rf_model.json") -> None:
        #Serialize key parameters
        import joblib
        joblib.dump(self.rf, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BusDelayRandomForest":
        import joblib
        obj = cls()
        obj.rf = joblib.load(path)
        return obj
