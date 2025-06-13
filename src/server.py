import flwr as fl
from flwr.server import Server
from flwr.server.history import History
from typing import Optional, Dict, List, Tuple
import json
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
import torch

from .client import CustomClient
from .data_utils import load_client_data
from .model import CustomFashionModel
from .strategy import FedAvgStrategy
from .fedmedian_strategy import FedMedian
from .krum_strategy import Krum
from .client_manager import CustomClientManager


def convert_metrics(metrics: Dict[str, List[tuple[int, float]]]) -> List[tuple[int, Dict[str, float]]]:
    converted: Dict[int, Dict[str, float]] = {}
    for metric_name, metric_list in metrics.items():
        for round_num, val in metric_list:
            if round_num not in converted:
                converted[round_num] = {}
            converted[round_num][metric_name] = float(val)
    return sorted(converted.items())

def save_results(history: History, filename: str = "results.json") -> None:
    results = {
        "losses_distributed": [(rnd, float(loss)) for rnd, loss in history.losses_distributed],
        "metrics_distributed": convert_metrics(getattr(history, "metrics_distributed", {})),
        "metrics_distributed_fit": convert_metrics(getattr(history, "metrics_distributed_fit", {})),
        "losses_centralized": [(rnd, float(loss)) for rnd, loss in getattr(history, "losses_centralized", [])],
        "metrics_centralized": convert_metrics(getattr(history, "metrics_centralized", {}))
    }

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {filename}")


def get_strategy(strategy_name: str = "fedavg") -> Strategy:
    """Return the aggregation strategy based on the input name."""
    if strategy_name == "fedavg":
        return FedAvgStrategy()
    elif strategy_name == "fedmedian":
        return FedMedian()
    elif strategy_name == "krum":
        return Krum()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def run_server(
    server_address: str = "127.0.0.1:8080",
    num_rounds: int = 3,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    strategy_name: str = "fedavg",
    output_file: str = "results.json"
) -> History:
    config = fl.server.ServerConfig(num_rounds=num_rounds)

    if strategy is None:
        strategy = get_strategy(strategy_name)

    print(f"Starting server for {num_rounds} rounds...")
    history = fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
        client_manager=client_manager
    )

    save_results(history, output_file)
    return history