from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import numpy as np
from flwr.common import (
    GetPropertiesIns, GetPropertiesRes,
    GetParametersIns, GetParametersRes,
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    Parameters, ndarrays_to_parameters,
    parameters_to_ndarrays, Scalar, Code, Status
)
from flwr.client import Client
from torch.utils.data import DataLoader
import random

class CustomClient(Client):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        attack_type: str = "none",  # "none", "data", or "model"
        poison_ratio: float = 0.5   # Ratio of labels to flip for data poisoning
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.attack_type = attack_type
        self.poison_ratio = poison_ratio

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={}
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = ndarrays_to_parameters(self.model.get_model_parameters())
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters
        )

    def _data_poisoning(self, target: torch.Tensor) -> torch.Tensor:
        """Implement label flipping attack"""
        # Flip labels for a portion of the data
        mask = torch.rand(len(target)) < self.poison_ratio
        poisoned_target = target.clone()
        poisoned_target[mask] = 9 - poisoned_target[mask]  # Flip to complementary class
        return poisoned_target

    def _model_poisoning(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Implement model poisoning by scaling updates"""
        # Scale up the parameters to dominate aggregation
        poisoned_parameters = [param * 3.0 for param in parameters]  # Scale by 3
        return poisoned_parameters

    def fit(self, ins: FitIns) -> FitRes:
        parameters = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(parameters)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Train normally (with potential data poisoning)
        loss, accuracy = self.model.train_epoch(
            self.train_loader, 
            criterion, 
            optimizer, 
            self.device,
            attack_type=self.attack_type,
            poison_fn=self._data_poisoning if self.attack_type == "data" else None
        )
        
        parameters = self.model.get_model_parameters()
        
        # Apply model poisoning if specified
        if self.attack_type == "model":
            parameters = self._model_poisoning(parameters)
        
        parameters_prime = ndarrays_to_parameters(parameters)
        
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters_prime,
            num_examples=len(self.train_loader.dataset),
            metrics={"train_loss": loss, "train_accuracy": accuracy}
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        parameters = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(parameters)
        
        criterion = nn.CrossEntropyLoss()
        loss, accuracy = self.model.test_epoch(
            self.test_loader, criterion, self.device
        )
        
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.test_loader.dataset),
            metrics={"val_accuracy": accuracy}
        )

    def to_client(self) -> 'CustomClient':
        return self