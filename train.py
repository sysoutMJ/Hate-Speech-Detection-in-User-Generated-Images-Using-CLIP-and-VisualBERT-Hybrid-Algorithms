from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
import numpy as np
import torch
from model import TrueHybridHateDetector
from dataset_dataloader import Config, create_dataloaders
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Implementation of Focal Loss for binary classification.

        Args:
            alpha (float): Weighting factor for the positive class (between 0-1).
                           Higher values give more importance to positive class.
            gamma (float): Focusing parameter that reduces loss for easy examples.
                           Higher values focus more on hard examples.
            reduction (str): 'mean', 'sum', or 'none' for the reduction method.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Calculate focal loss.

        Args:
            inputs (Tensor): Raw model outputs (before sigmoid)
            targets (Tensor): Target labels (0 or 1)

        Returns:
            Tensor: Computed focal loss
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # For numerical stability, use built-in binary cross entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate focal term: (1 - pt)^gamma
        # For positive samples (y=1): pt = p
        # For negative samples (y=0): pt = 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        # For positive samples: use alpha
        # For negative samples: use (1-alpha)
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Combine everything
        focal_loss = alpha_weight * focal_term * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def validate_no_training(self, dataloader):
    """
    Evaluate model performance without training.
    """
    self.eval()  # Set model to evaluation mode
    all_outputs = []  # Store raw outputs for threshold tuning
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch["labels"].to(self.device)

            # Get raw outputs (before sigmoid)
            outputs = self.forward(batch["text"], batch["image_paths"])

            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)

    # Threshold optimization
    thresholds = np.linspace(0.3, 0.7, 20)
    best_thresh = self.config.desired_threshold  # Start with current threshold
    best_f1 = 0

    for thresh in thresholds:
        preds = (torch.sigmoid(torch.tensor(all_outputs)) >= thresh).int()
        current_f1 = f1_score(all_labels, preds, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh

    print(f"Optimal threshold: {best_thresh:.4f} (F1={best_f1:.4f})")

    # Use best threshold for final predictions
    final_preds = (torch.sigmoid(torch.tensor(all_outputs)) >= best_thresh).int()

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, final_preds),
        "precision": precision_score(all_labels, final_preds, zero_division=0),
        "recall": recall_score(all_labels, final_preds, zero_division=0),
        "f1": best_f1,
        "confusion_matrix": confusion_matrix(all_labels, final_preds).tolist(),
        "predictions": final_preds.numpy(),
        "labels": all_labels,
        "optimal_threshold": best_thresh,
    }

    return metrics


def train_model(
    self,
    train_loader,
    val_loader,
    pos_weight,
    epochs=5,
    lr=5e-5,
    patience=3,
    min_lr=1e-6,
    weight_decay=0.05,
    T_0=5,
    clip_grad_norm_val=1.0,
    model_save_path="best_model.pth",
    use_focal_loss=True,
    focal_alpha=0.25,
    focal_gamma=2.0,
):
    """
    Enhanced training method with:
    - Focal Loss
    - Learning rate scheduling
    - Mixed precision training
    - Early stopping
    - Detailed logging
    - Validation loss tracking
    - Dynamic threshold optimization
    """
    optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
    if use_focal_loss:
        # FocalLoss
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(self.device)
            if pos_weight
            else None
        )
        print(f"Using BCE Loss (pos_weight={pos_weight})")

    scaler = torch.GradScaler(str(self.device))

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_metrics": [],
        "learning_rates": [],
        "best_f1": 0,
        "best_epoch": 0,
        "best_val_loss": float("inf"),
    }
    epochs_no_improve = 0

    warmup_epochs = 3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        eta_min=min_lr,
    )

    for epoch in range(epochs):
        # Training phase
        self.train()

        # LR warmup
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * lr_scale
        else:
            scheduler.step()

        epoch_loss = 0.0

        print(f"\n{'=' * 30}")
        print(f" Epoch {epoch + 1}/{epochs}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'=' * 30}")

        with tqdm(train_loader, unit="batch", desc="Training") as tepoch:
            for batch in tepoch:
                optimizer.zero_grad()
                labels = batch["labels"].float().to(self.device)

                with torch.autocast(str(self.device)):
                    logits = self(batch["text"], batch["image_paths"])
                    if logits.dim() == 0:
                        logits = logits.unsqueeze(0)
                    if labels.dim() == 0:
                        labels = labels.unsqueeze(0)

                    logits = logits.view(-1)
                    labels = labels.view(-1)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad_norm_val
                )  # Clip to 1.0 This is Gradient Clipping
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation phase
        val_metrics, val_loss = validate_with_loss(self, val_loader, criterion)
        self.config.desired_threshold = val_metrics["optimal_threshold"]

        # Track current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        # Formatted summary printout
        print(f"\n--- Epoch {epoch + 1} Summary ---")
        print(f"Train Loss     : {avg_train_loss:.4f}")
        print(f"Val Loss       : {val_loss:.4f}")
        print(f"Learning Rate  : {current_lr:.2e}")
        print(f"Val Accuracy   : {val_metrics['accuracy']:.4f}")
        print(f"Val Precision  : {val_metrics['precision']:.4f}")
        print(f"Val Recall     : {val_metrics['recall']:.4f}")
        print(f"Val F1 Score   : {val_metrics['f1']:.4f}")
        print(f"Opt Threshold  : {val_metrics['optimal_threshold']:.4f}")
        print(
            f"Confusion Matrix:\n{np.array2string(np.array(val_metrics['confusion_matrix']), formatter={'int': lambda x: f'{x:4d}'})}"
        )

        # Model checkpointing based on validation loss
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            history["best_f1"] = val_metrics["f1"]

            torch.save(self.state_dict(), model_save_path)
            print(f"\n✅ New best model saved (Val Loss = {val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⏸️ No improvement for {epochs_no_improve}/{patience} epoch(s)")

            if epochs_no_improve >= patience:
                print("\n⛔ Early stopping triggered.")
                print(
                    f"🏁 Best Val Loss: {history['best_val_loss']:.4f} (Epoch {history['best_epoch'] + 1})"
                )
                break

        # Additional early stopping if learning rate becomes too small
        if current_lr <= min_lr:
            print("\n⛔ Minimum learning rate reached - stopping training.")
            break

    # Load best model
    self.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    print(
        f"\n✅ Training complete. Best Val Loss: {history['best_val_loss']:.4f} (Epoch {history['best_epoch'] + 1})"
    )
    print(f"Final learning rate: {history['learning_rates'][-1]:.2e}")

    return history


def validate_with_loss(self, dataloader, criterion):
    """Enhanced validation that returns both metrics and loss"""
    self.eval()
    all_outputs = []
    all_labels = []
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            labels = batch["labels"].float().to(self.device)
            outputs = self(batch["text"], batch["image_paths"])

            # Calculate validation loss
            # loss = criterion(outputs, labels)
            loss = criterion(outputs.view(-1), labels.view(-1))

            val_loss += loss.item()

            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(dataloader)
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)

    # Threshold optimization
    thresholds = np.linspace(0.3, 0.7, 20)
    best_thresh = self.config.desired_threshold
    best_f1 = 0

    for thresh in thresholds:
        preds = (torch.sigmoid(torch.tensor(all_outputs)) >= thresh).int()
        current_f1 = f1_score(all_labels, preds, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh

    final_preds = (torch.sigmoid(torch.tensor(all_outputs)) >= best_thresh).int()

    metrics = {
        "accuracy": accuracy_score(all_labels, final_preds),
        "precision": precision_score(all_labels, final_preds, zero_division=0),
        "recall": recall_score(all_labels, final_preds, zero_division=0),
        "f1": best_f1,
        "confusion_matrix": confusion_matrix(all_labels, final_preds).tolist(),
        "optimal_threshold": best_thresh,
    }

    return metrics, val_loss


def print_metrics(metrics):
    """Pretty-print the evaluation metrics"""
    print("\nValidation Metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(
        np.array2string(
            np.array(metrics["confusion_matrix"]),
            formatter={"int": lambda x: f"{x:4d}"},
        )
    )


def evaluate_saved_model(model_path, model, val_loader=None, test_loader=None):
    """Evaluate a saved model on validation and test sets."""
    # Load the best model
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    if val_loader:
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_metrics = validate_no_training(model, val_loader)
        print_metrics(val_metrics)
        return val_metrics

    # Optionally evaluate on test set
    if test_loader:
        print("\nEvaluating on test set...")
        test_metrics = validate_no_training(model, test_loader)
        print("\nTest Set Performance:")
        print_metrics(test_metrics)
        return test_metrics


if __name__ == "__main__":
    # Initialize config and model
    # This for both model and datasetdataloader
    config = Config()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.desired_threshold = 0.5  # For model
    config.batch_size = 32
    config.train_dataset_path = r"K:\0505\train_70_contains_10_only.jsonl"
    config.val_dataset_path = r"K:\0505\val_15.jsonl"
    config.test_dataset_path = r"K:\0505\test_15_stratified_to_50_50_757.jsonl"

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Initialize model
    model = TrueHybridHateDetector(config).to(config.device)

    # Train the model ---------------------------------------------------

    # history = train_model(
    #     model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     pos_weight=1.47,  # For class imbalance. Since focal loss is applied, this will be applied if use_focal_loss is set to FALSE
    #     epochs=100,
    #     lr=2e-5,  # Change later to 3e-5
    #     patience=5,  # For early stopiing
    #     min_lr=1e-7,  # For scheduler, eta_min
    #     weight_decay=0.1,
    #     T_0=5,  # For CosineAnnealingWarmRestarts, scheduler
    #     clip_grad_norm_val=2.0,  # For Relax Gradient Clipping
    #     model_save_path="0506_best_model.pth",
    #     use_focal_loss=True,
    #     focal_alpha=0.595,  # Weight for positive class (adjust based on class imbalance)
    #     focal_gamma=2.0,  # Focusing parameter (higher = more focus on hard examples)
    # )

    # plt.plot(history["train_loss"], label="Train Loss")
    # plt.plot(history["val_loss"], label="Validation Loss")
    # plt.legend()
    # plt.show()

    # -------------------------------------------------------------------

    # TO BE USED LATER
    # ⚠️⚠️⚠️
    # Evaluate on validation set
    evaluate_saved_model(
        r"K:\0505\lr_2e-5_1.1_0506_best_model.pth",
        model,
        val_loader=None,
        test_loader=test_loader,
    )

    # ⚠️⚠️⚠️
    # Optionally evaluate on test set
    # if test_loader:
    #     test_metrics = validate_no_training(model, test_loader)
    #     print("\nTest Set Performance:")
    #     print_metrics(test_metrics)
