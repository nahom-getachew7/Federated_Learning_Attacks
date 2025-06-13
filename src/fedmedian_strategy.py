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

class FedMedian(Strategy):
    def __init__(self, fraction_fit: float = 1.0, fraction_evaluate: float = 1.0):
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.robust_scale = 0.1 

    def initialize_parameters(self, client_manager):
        # Initialize with random parameters
        model = CustomFashionModel()
        return ndarrays_to_parameters(model.get_model_parameters())

    def configure_fit(self, server_round, parameters, client_manager):
        # Sample all available clients for fit
        clients = client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=1
        )
        
        config = {
            "server_round": server_round,
            "epochs": 3,
            "batch_size": 64,
            "learning_rate": 0.01
        }
        
        fit_ins = FitIns(parameters=parameters, config=config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        
        # Add parameter clipping
        clipped_weights = []
        for client_weights in weights:
            clipped = [
                np.clip(w, -self.robust_scale, self.robust_scale) 
                for w in client_weights
            ]
            clipped_weights.append(clipped)
        
        # Compute coordinate-wise median
        median_weights = [
            np.median(np.stack([w[i] for w in clipped_weights]), axis=0)
            for i in range(len(clipped_weights[0]))
        ]
        
        # Compute metrics
        avg_loss = np.mean([res.metrics["train_loss"] for _, res in results])
        avg_acc = np.mean([res.metrics["train_accuracy"] for _, res in results])

        return ndarrays_to_parameters(median_weights), {
            "train_loss": float(avg_loss),
            "train_accuracy": float(avg_acc)
        }

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Evaluate on all available clients
        clients = client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=1
        )
        evaluate_ins = EvaluateIns(parameters=parameters, config={})
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        # Aggregate evaluation metrics
        total_examples = sum([res.num_examples for _, res in results])
        weighted_loss = sum([res.loss * res.num_examples for _, res in results]) / total_examples
        avg_accuracy = np.mean([res.metrics["val_accuracy"] for _, res in results])

        return float(weighted_loss), {
            "val_loss": float(weighted_loss),
            "val_accuracy": float(avg_accuracy)
        }
    