import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import argparse
from rtpt import RTPT

from mbg.scorer.context_proximity_dataset import ContextContourDataset, context_collate_fn
from mbg.scorer.context_contour_scorer import ContextContourScorer
from mbg.scorer import scorer_config


def train_model(args, principle, input_type, sample_size, device, log_wandb=True, n=100, epochs=10, data_num=100000):
    # Resolve paths
    # path_map = {
    #     "closure": (scorer_config.get_data_path(args.remote, principle), scorer_config.get_model_file_name(args.remote, principle)),
    #     "proximity": (scorer_config.get_data_path(args.remote, principle), scorer_config.get_model_file_name(args.remote, principle)),
    #     "continuity": (scorer_config.get_data_path(args.remote, principle), scorer_config.get_model_file_name(args.remote, principle)),
    #     "symmetry": (scorer_config.get_data_path(args.remote, principle), scorer_config.get_model_file_name(args.remote, principle)),
    #     "similarity": (scorer_config.get_data_path(args.remote, principle), scorer_config.get_model_file_name(args.remote, principle)),
    # }
    # if principle not in path_map:
    #     raise ValueError(f"Unsupported principle: {principle}")
    data_path = scorer_config.get_data_path(args.remote, principle) / "train"
    model_path = scorer_config.get_model_file_name(args.remote, principle)
    model_path_best = str(model_path).replace(".pt", "_best.pt")
    model_path_latest = str(model_path).replace(".pt", "_latest.pt")

    # Input dimension
    input_dim_map = {"pos": 2, "pos_color": 5, "pos_color_size": 7, "color_size": 5}
    if input_type not in input_dim_map:
        raise ValueError(f"Unsupported input type: {input_type}")
    input_dim = input_dim_map[input_type]

    # Setup
    model = ContextContourScorer(input_dim=input_dim).to(device)
    orders = list(range(n))
    random.shuffle(orders)  # Randomly shuffle task orders
    train_dataset = ContextContourDataset(data_path, orders, input_type, sample_size, device=device, data_num=data_num, task_num=n, remove_cache=args.remove_cache)
    test_dataset = ContextContourDataset(data_path, orders, input_type, sample_size, device=device, data_num=data_num, split="test", task_num=n)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=context_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=context_collate_fn)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for ci, cj, ctx, label in train_loader:
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
        avg_acc = acc
        print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

        # Test loop
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for ci, cj, ctx, label in test_loader:
                ci, cj = ci.to(device), cj.to(device)
                label = label.to(device)
                ctx_tensor = ctx[0].unsqueeze(0).to(device)
                logits = model(ci, cj, ctx_tensor)
                loss = criterion(logits, label)
                test_loss += loss.item() * len(label)
                pred = (torch.sigmoid(logits) > 0.5).float()
                test_correct += (pred == label).sum().item()
                test_total += len(label)
        test_acc = test_correct / test_total if test_total > 0 else 0
        test_avg_loss = test_loss / test_total if test_total > 0 else 0
        print(f"[Epoch {epoch + 1}] Test Loss: {test_avg_loss:.4f} | Test Acc: {test_acc:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": acc,
            "test_loss": test_avg_loss,
            "test_accuracy": test_acc
        })
        # Save latest model
        torch.save(model.state_dict(), model_path_latest)
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path_best)
    print(f"Model saved to {model_path}")
    return test_acc, test_avg_loss


def parse_device(device_str):
    if device_str.isdigit():
        return f"cuda:{device_str}"
    elif device_str.startswith("cuda") or device_str == "cpu":
        return device_str
    else:
        raise ValueError(f"Invalid device string: {device_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sample_size_list", type=str, default="5,10,20,50,100")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--principle", type=str)
    parser.add_argument("--input_types", type=str, default="pos_color_size")
    parser.add_argument("--data_nums", type=str, default="10000,50000,100000", )
    parser.add_argument("--remove_cache", action="store_true", help="Remove existing cache files before processing")
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()
    args.device = parse_device(args.device)
    input_type = args.input_types

    data_num_list = [int(x) for x in args.data_nums.split(",")]
    sample_size_list = [int(x) for x in args.sample_size_list.split(",")]
    report = []
    p = args.principle
    rtpt = RTPT(name_initials='JIS', experiment_name=f'GRM-Grp-{args.principle}', max_iterations=1)
    rtpt.start()
    for data_num in data_num_list:
        for sample_size in sample_size_list:
            wandb.init(project=f"grp-{args.principle}", config={"epochs": args.epochs, "batch_size": 1, "learning_rate": 1e-3,
                                                                "sample_size": sample_size, "device": args.device,
                                                                "input_type": input_type, "data_num": data_num},
                       name=f"s{sample_size}_n{args.n}_d{data_num}_ep{args.epochs}")

            print(f"\n=== Training {p} with {input_type} ===")
            acc, loss = train_model(args, p, input_type, sample_size, args.device, log_wandb=True, n=args.n, epochs=args.epochs, data_num=data_num)
            report.append((p, input_type, data_num, sample_size, acc, loss))
            wandb.finish()

    # Final report
    print("\n==== Final Report ====")
    print(f"{'Principle':<12} {'Input Type':<16} {'Data Num':>10} {'Sample Size':>10} {'Accuracy':>10} {'Loss':>10}")
    print("-" * 62)
    for p, input_type, data_num, sample_size, a, l in report:
        print(f"{p:<12} {input_type:<16} {data_num:>10} {sample_size:>10} {a:>10.4f} {l:>10.4f}")


if __name__ == "__main__":
    main()
