# Created by MacBook Pro at 03.06.25

import torch
import torch.nn as nn


from deepproblog.model import Model
from deepproblog.train import train_model
from deepproblog.query import Query
from deepproblog.dataset import QueryDataset
from deepproblog.engines import ExactEngine
from problog.logic import Term, Var
from deepproblog.dataset import Dataset


class ConfidenceCalibrator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def train_from_data(self, X, y, device, lr=0.01, epochs=100, verbose=True, ):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()

        loss_history = []

        for epoch in range(epochs):
            opt.zero_grad()
            pred = self.forward(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            loss_history.append(loss.item())

            # if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            #     print(f"[Epoch {epoch + 1:3d}] Loss: {loss.item():.4f}")

        # Final diagnostic
        # if verbose:
        #     print(f"\nInitial loss: {loss_history[0]:.4f} | Final loss: {loss_history[-1]:.4f}")
        #     if loss_history[-1] > loss_history[0]:
        #         print("⚠️ Loss increased — potential overfitting or bad learning rate.")
        #     elif abs(loss_history[-1] - loss_history[0]) < 1e-3:
        #         print("⚠️ Loss did not change much — may be underfitting.")
        #     else:
        #         print("✅ Loss decreased — model likely learned meaningful signal.")


from deepproblog.train import StopCondition

class FixedEpochStopCondition(StopCondition):
    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs

    def is_stop(self, train_object):
        return train_object.epoch >= self.max_epochs
    
# Custom dataset that returns single-item batches
class InMemoryDataset(Dataset):
    def __init__(self, queries, facts_list):
        self.queries = queries
        self.facts_list = facts_list

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        # Return (query, facts) tuple for DeepProbLog
        return (self.queries[index], self.facts_list[index])


    def to_query(self, i):
        # Return the Query object for index i
        return self.queries[i]
# Main DeepProbLog calibrator wrapper
class DeepProbLogCalibrator:
    def __init__(self, rules_path: str, facts: list = None):
        # facts: list of grounded fact strings, or None
        self.rules_path = rules_path
        self.facts = facts if facts is not None else []
        # Combine rules and facts into a single logic program
        with open(rules_path, "r") as f:
            rules_str = f.read()
        program_str = rules_str + "\n" + "\n".join(self.facts)
        # Write combined program to a temporary file
        import tempfile
        self.program_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".pl", delete=False)
        self.program_file.write(program_str)
        self.program_file.flush()
        self.model = Model(self.program_file.name, networks={})
        self.model.set_engine(ExactEngine(self.model))

    def train(self, dataset):
        stop_condition = FixedEpochStopCondition(max_epochs=10)
        train_model(self.model, dataset, stop_condition=stop_condition)

    def predict(self, facts):
        # For prediction, create a temporary program with rules + new facts
        with open(self.rules_path, "r") as f:
            rules_str = f.read()
        program_str = rules_str + "\n" + "\n".join(facts)
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".pl", delete=False) as temp_file:
            temp_file.write(program_str)
            temp_file.flush()
            model = Model(temp_file.name, networks={})
            model.set_engine(ExactEngine(model))
            Y = Var("Y")
            query_term = Term("label", Term("img"), Y)
            query = Query(query_term, p=1.0)
            result = model.solve([query])
            return float(result[0].value)