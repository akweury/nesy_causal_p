import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse

from mbg.scorer.context_proximity_dataset import ContextContourDataset, context_collate_fn
from mbg.scorer.context_contour_scorer import ContextContourScorer
from mbg.scorer import scorer_config


def train_model(principle, input_type, device, log_wandb=True):
    # Resolve paths
    path_map = {
        "closure": (scorer_config.closure_path, scorer_config.CLOSURE_MODEL),
        "proximity": (scorer_config.proximity_path, scorer_config.PROXIMITY_MODEL),
        "continuity": (scorer_config.continuity_path, scorer_config.CONTINUITY_MODEL),
        "symmetry": (scorer_config.symmetry_path, scorer_config.SYMMETRY_MODEL),
        "similarity": (scorer_config.SIMILARITY_PATH, scorer_config.SIMILARITY_MODEL),
    }
    if principle not in path_map:
        raise ValueError(f"Unsupported principle: {principle}")
    data_path, model_path = path_map[principle]

    # Input dimension
    input_dim_map = {
        "pos": 2,
        "pos_color": 5,
        "pos_color_size": 7,
    }
    if input_type not in input_dim_map:
        raise ValueError(f"Unsupported input type: {input_type}")
    input_dim = input_dim_map[input_type]

    # Setup
    model = ContextContourScorer(input_dim=input_dim).to(device)
    dataset = ContextContourDataset(data_path, input_type, data_num=10000, task_num=100)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=context_collate_fn)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if log_wandb:
        wandb.init(project="grb-context-train", name=f"{principle}_{input_type}", config={
            "principle": principle,
            "input_type": input_type,
            "epochs": 50,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "device": device,
        })

    for epoch in range(50):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for ci, cj, ctx, label in tqdm(data_loader, desc=f"{principle}-{input_type}"):
            ci, cj = ci.to(device), cj.to(device)
            label = label.to(device)
            ctx_tensor = ctx[0].unsqueeze(0).to(device)

            logits = model(ci, cj, ctx_tensor)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(label)
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == label).sum().item()
            total += len(label)

        acc = correct / total
        avg_loss = total_loss / total
        print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

        if log_wandb:
            wandb.log({"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc})

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    if log_wandb:
        wandb.finish()


def parse_device(device_str):
    if device_str.isdigit():
        return f"cuda:{device_str}"
    elif device_str.startswith("cuda") or device_str == "cpu":
        return device_str
    else:
        raise ValueError(f"Invalid device string: {device_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--all", action="store_true", help="Train all principles and input types")
    args = parser.parse_args()
    args.device = parse_device(args.device)
    principles = ["closure", "proximity", "continuity", "symmetry", "similarity"]
    input_types = ["pos", "pos_color", "pos_color_size"]

    report = []
    if args.all:
        for p in principles:
            for t in input_types:
                print(f"\n=== Training {p} with {t} ===")
                acc, loss = train_model(p, t, args.device, log_wandb=True)
                report.append((p, t, acc, loss))
    else:
        acc, loss = train_model("similarity", "pos_color", args.device, log_wandb=True)
        report.append(("similarity", "pos_color", acc, loss))

    # Final report
    print("\n==== Final Report ====")
    print(f"{'Principle':<12} {'Input Type':<16} {'Accuracy':>10} {'Loss':>10}")
    print("-" * 52)
    for p, t, a, l in report:
        print(f"{p:<12} {t:<16} {a:>10.4f} {l:>10.4f}")
