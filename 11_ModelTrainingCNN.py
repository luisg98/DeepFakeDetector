import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
from PIL import ImageFile
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report
import wandb

# ====================================
# CONFIGURATION
# ====================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = 'checkpoints/CNN'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'model_checkpoint.pth')
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'model_final.pth')
METRICS_PATH = os.path.join(CHECKPOINT_DIR, 'metrics.json')
CLASS_REPORT_DIR = os.path.join(CHECKPOINT_DIR, 'class_reports')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CLASS_REPORT_DIR, exist_ok=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Initialize Weights & Biases
wandb.init(project="CNN-Training", config={
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "architecture": "EfficientNetB0"
})

# ====================================
# MODEL DEFINITION
# ====================================
def build_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    for name, param in model.named_parameters():
        if "features.7" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model.to(DEVICE)

# ====================================
# EVALUATION FUNCTION
# ====================================
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# ====================================
# CLASSIFICATION REPORT FUNCTION
# ====================================
def get_classification_report(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(DEVICE)
            preds = torch.argmax(model(imgs), dim=1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
    return classification_report(y_true, y_pred, output_dict=True)

# ====================================
# MAIN EXECUTION
# ====================================
if __name__ == "__main__":
    # Load datasets
    train_dataset = torch.load('data/processed/train_dataset.pt', weights_only=False)
    val_dataset = torch.load('data/processed/val_dataset.pt', weights_only=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize model, optimizer, scheduler
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    start_epoch = 0
    best_acc = 0
    patience = 3
    epochs_no_improve = 0

    # Resume from checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")

    # Metrics dictionary
    metrics = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Log to Weights & Biases
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        # Save classification report
        report = get_classification_report(model, val_loader)
        with open(os.path.join(CLASS_REPORT_DIR, f"epoch_{epoch}_report.json"), 'w') as f:
            json.dump(report, f, indent=2)

        # Early stopping & checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping triggered")
                break

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, CHECKPOINT_PATH)
        scheduler.step()

    # Save metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)
    print(f"Final model and training metrics saved.")

    # Finish WandB run
    wandb.finish()
