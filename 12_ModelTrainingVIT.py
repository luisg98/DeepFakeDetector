import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm import create_model
from tqdm import tqdm
from PIL import ImageFile
from sklearn.metrics import classification_report

# Fix for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================================
# CONFIGURATION
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 3

CHECKPOINT_PATH = 'checkpoints/DeiT/deit_checkpoint.pth'
FINAL_MODEL_PATH = 'checkpoints/DeiT/deit_final.pth'
METRICS_PATH = 'checkpoints/DeiT/deit_metrics.json'
REPORTS_DIR = 'checkpoints/DeiT/reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

# ================================
# MODEL DEFINITION
# ================================
def build_model():
    model = create_model("deit_tiny_patch16_224", pretrained=True, num_classes=2)
    return model.to(DEVICE)

# ================================
# EVALUATION FUNCTION
# ================================
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_labels, all_preds

# ================================
# MAIN EXECUTION
# ================================
if __name__ == "__main__":
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    # Load datasets
    train_dataset = torch.load('data/processed/train_dataset.pt', weights_only=False)
    val_dataset = torch.load('data/processed/val_dataset.pt', weights_only=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    metrics = {"train_loss": [], "val_loss": [], "val_acc": []}

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} - Training (DeiT)"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion)

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save per-epoch classification report
        report_dict = classification_report(y_true, y_pred, target_names=["fake", "real"], output_dict=True)
        with open(os.path.join(REPORTS_DIR, f"epoch_{epoch}_report.json"), "w") as f:
            json.dump(report_dict, f, indent=4)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved to: {CHECKPOINT_PATH}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Save final model and metrics
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Final model saved to: {FINAL_MODEL_PATH}")

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)
    print(f"Training metrics saved to: {METRICS_PATH}")
