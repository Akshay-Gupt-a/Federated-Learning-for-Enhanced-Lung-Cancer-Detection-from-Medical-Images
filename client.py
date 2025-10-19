import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import flwr as fl
from common import build_or_load_model, load_client_data


# ------------------- Helper Functions -------------------

def save_results(client_dir, round_num, phase, loss, acc, auc_score):
    """Append results to CSV file"""
    results_file = os.path.join(client_dir, "client_results.csv")
    file_exists = os.path.isfile(results_file)

    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Round", "Phase", "Loss", "Accuracy", "AUC"])
        writer.writerow([round_num, phase, loss, acc, auc_score])


def plot_metrics(client_dir, round_num, y_true, y_pred, y_prob):
    """Save ROC curve and Confusion Matrix plots"""
    results_dir = os.path.join(client_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Round {round_num}")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, f"roc_round{round_num}.png"))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Cancer"],
        yticklabels=["Normal", "Cancer"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Round {round_num}")
    plt.savefig(os.path.join(results_dir, f"conf_matrix_round{round_num}.png"))
    plt.close()


# ------------------- Flower Client -------------------

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_ds, val_ds, local_epochs, client_dir):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.local_epochs = local_epochs
        self.client_dir = client_dir

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Update global weights
        self.model.set_weights(parameters)

        # Train locally
        self.model.fit(
            self.train_ds,
            epochs=self.local_epochs,
            verbose=1
        )

        # Local eval after training
        loss, acc, auc_score = self.model.evaluate(self.val_ds, verbose=0)
        print(f"📊 after_fit - loss: {loss:.4f}, acc: {acc:.4f}, auc: {auc_score:.4f}")

        # Save results
        round_num = config.get("round", 0)
        save_results(self.client_dir, round_num, "fit", loss, acc, auc_score)

        return self.model.get_weights(), len(self.train_ds), {}

    def evaluate(self, parameters, config):
        # Update with global weights
        self.model.set_weights(parameters)

        y_true, y_pred, y_prob = [], [], []

        for x, y in self.val_ds:
            probs = self.model.predict(x, verbose=0).ravel()
            preds = (probs > 0.5).astype(int)
            y_true.extend(y.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        # Compute metrics
        loss, acc, auc_score = self.model.evaluate(self.val_ds, verbose=0)
        print(f"📊 after_eval - loss: {loss:.4f}, acc: {acc:.4f}, auc: {auc_score:.4f}")

        # Save results
        round_num = config.get("round", 0)
        save_results(self.client_dir, round_num, "eval", loss, acc, auc_score)

        # Save ROC & Confusion Matrix
        plot_metrics(self.client_dir, round_num, y_true, y_pred, y_prob)

        return float(loss), len(self.val_ds), {"accuracy": float(acc), "auc": float(auc_score)}


# ------------------- Main -------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_dir", type=str, required=True,
                        help="Path to federated_clients/client_X directory")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--local_epochs", type=int, default=1)
    args = parser.parse_args()

    # Build/load model
    model = build_or_load_model()

    # Load train and validation datasets
    train_ds, _ = load_client_data(args.client_dir, shuffle=True)
    val_ds, _ = load_client_data(args.client_dir, shuffle=False)

    # Start client
    client = FlowerClient(model, train_ds, val_ds, args.local_epochs, args.client_dir)
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client()
    )
