# Created by MacBook Pro at 16.04.25
import os
import numpy as np

import config


def load_embeddings_with_labels(embedding_dir):
    shape_classes = ["triangle", "rectangle", "ellipse"]
    all_embeddings = []
    all_labels = []

    for shape in shape_classes:
        path = os.path.join(embedding_dir, f"{shape}_embeddings.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found: {path}")

        embeddings = np.load(path)
        labels = [shape] * len(embeddings)

        all_embeddings.append(embeddings)
        all_labels.extend(labels)

    return np.vstack(all_embeddings), np.array(all_labels)


def save_memory_bank(embedding_dir, output_path=None):
    embeddings, labels = load_embeddings_with_labels(embedding_dir)
    np.savez_compressed(output_path / "feature_closure_memory_bank.npz", embeddings=embeddings, labels=labels)
    print(f"Memory bank saved to: {output_path}")
    print(f"Total entries: {len(labels)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    embedding_dir = config.mb_outlines
    output_path = config.mb_outlines
    args = parser.parse_args()

    save_memory_bank(embedding_dir, output_path)
