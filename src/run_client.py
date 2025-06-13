import argparse
from .data_utils import load_client_data
from .model import CustomFashionModel
from .client import CustomClient
import torch
import flwr as fl

def run_client(cid: int, attack_type: str = "none", poison_ratio: float = 0.5) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = load_client_data(
        cid=cid,
        data_dir="./data/client_data",
        batch_size=64
    )
    
    model = CustomFashionModel().to(device)
    client = CustomClient(model, train_loader, val_loader, device, attack_type, poison_ratio)
    
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FL client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    parser.add_argument("--attack-type", type=str, default="none", 
                       choices=["none", "data", "model"], help="Type of attack to perform")
    parser.add_argument("--poison-ratio", type=float, default=0.5, 
                       help="Ratio of labels to flip for data poisoning")
    args = parser.parse_args()
    
    print(f"Starting client {args.cid} with attack type: {args.attack_type}")
    run_client(args.cid, args.attack_type, args.poison_ratio)