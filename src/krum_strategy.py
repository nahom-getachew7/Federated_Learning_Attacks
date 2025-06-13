from flwr.server.strategy import Strategy
from flwr.common import (
    Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
    Scalar
)
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from .model import CustomFashionModel
from scipy.spatial.distance import cdist


class Krum(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        byzantine_clients: int = 1,
    ):
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.byzantine_clients = byzantine_clients

    def initialize_parameters(self, client_manager):
        # NOTE: Replace with external or dynamic model loader if possible
        model = CustomFashionModel()
        return ndarrays_to_parameters(model.get_model_parameters())

    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=1,
        )

        config = {
            "server_round": server_round,
            "epochs": 3,
            "batch_size": 64,
            "learning_rate": 0.01,
        }

        fit_ins = FitIns(parameters=parameters, config=config)
        return [(client, fit_ins) for client in clients]

    def _compute_krum_score(self, updates: np.ndarray) -> int:
        """Compute Krum scores and return index of most central update"""
        n = len(updates)
        f = self.byzantine_clients

        if n <= f + 2:
            raise ValueError(f"Krum requires > {f + 2} updates, got {n}")

        distances = cdist(updates, updates, metric="euclidean")
        scores = []

        for i in range(n):
            # Exclude self-distance before sorting
            sorted_dist = np.sort(np.delete(distances[i], i))
            score = np.sum(sorted_dist[:n - f - 2])
            scores.append(score)

        return int(np.argmin(scores))  # Index of client with lowest Krum score

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Convert to list of model weights (list of list of np.ndarray)
        weights = [parameters_to_ndarrays(res.parameters) for _, res in results]

        # Flatten weights into 1D vectors for distance comparison
        flattened_updates = [np.concatenate([w.flatten() for w in weight]) for weight in weights]
        updates_array = np.stack(flattened_updates)

        # Krum selection
        selected_idx = self._compute_krum_score(updates_array)
        selected_weights = weights[selected_idx]

        # Robust metric aggregation with fallback
        losses = [res.metrics.get("train_loss", 0.0) for _, res in results]
        accs = [res.metrics.get("train_accuracy", 0.0) for _, res in results]
        avg_loss = float(np.mean(losses))
        avg_acc = float(np.mean(accs))

        return ndarrays_to_parameters(selected_weights), {
            "train_loss": avg_loss,
            "train_accuracy": avg_acc,
        }

    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=1,
        )
        evaluate_ins = EvaluateIns(parameters=parameters, config={})
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_examples = sum(res.num_examples for _, res in results)
        weighted_loss = sum(res.loss * res.num_examples for _, res in results) / total_examples
        weighted_acc = (
            sum(res.metrics.get("val_accuracy", 0.0) * res.num_examples for _, res in results)
            / total_examples
        )

        return float(weighted_loss), {
            "val_loss": float(weighted_loss),
            "val_accuracy": float(weighted_acc),
        }
