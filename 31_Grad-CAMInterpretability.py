import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Fix for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =====================================
# CONFIGURATION
# =====================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
NUM_WORKERS = 2
CNN_MODEL_PATH = "checkpoints/CNN/model_final.pth"
DEIT_MODEL_PATH = "checkpoints/DeiT/deit_final.pth"
TESTSET_PATH = "data/processed/test_dataset.pt"
OUTPUT_DIR = "output/gradcam"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# LOAD MODELS
# =====================================
def load_cnn_model():
    # Try to determine the correct architecture from the checkpoint
    checkpoint = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
    
    # Check if it's just the state_dict or full checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Determine architecture based on keys
    keys = list(state_dict.keys())
    
    if any('features.' in key for key in keys):
        # This appears to be EfficientNet based on the key structure
        print("Detected EfficientNet architecture from checkpoint")
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    else:
        # Fallback to ResNet
        print("Using ResNet18 architecture")
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    # Load with weights_only=True for security
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()

def load_deit_model():
    from timm import create_model
    model = create_model("deit_tiny_patch16_224", pretrained=False, num_classes=2)
    # Load with weights_only=True for security
    model.load_state_dict(torch.load(DEIT_MODEL_PATH, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()

def reshape_transform(tensor, height=14, width=14):
    # Remove CLS token and reshape to (B, C, H, W)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, -1)
    result = result.permute(0, 3, 1, 2)
    return result

# =====================================
# GRADCAM FUNCTION
# =====================================
def apply_gradcam(model, image_tensor, target_layer, target_class, model_name, idx, is_vit=False):
    if is_vit:
        cam = GradCAM(
            model=model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform
        )
    else:
        cam = GradCAM(model=model, target_layers=[target_layer])

    # Ensure input is 4D [B, C, H, W]
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(target_class)])[0]

    image_np = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 0.5) + 0.5
    image_np = np.clip(image_np, 0, 1)

    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
    output_path = os.path.join(OUTPUT_DIR, f"gradcam_{model_name}_{idx}.png")
    Image.fromarray(cam_image).save(output_path)
    print(f"Saved Grad-CAM: {output_path}")

def get_target_layer(model, model_name):
    """Get the appropriate target layer for Grad-CAM based on model architecture"""
    if hasattr(model, 'features'):  # EfficientNet
        return model.features[-1]
    elif hasattr(model, 'layer4'):  # ResNet
        return model.layer4[-1]
    elif hasattr(model, 'blocks'):  # Vision Transformer
        return model.blocks[-1].norm1
    else:
        # Fallback - try to find the last convolutional layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError(f"Could not determine target layer for {model_name}")

# =====================================
# MAIN EXECUTION
# =====================================
if __name__ == "__main__":
    try:
        # Load models
        cnn_model = load_cnn_model()
        deit_model = load_deit_model()

        # Get target layers dynamically
        cnn_target_layer = get_target_layer(cnn_model, "CNN")
        deit_target_layer = get_target_layer(deit_model, "DeiT")

        print(f"CNN target layer: {cnn_target_layer}")
        print(f"DeiT target layer: {deit_target_layer}")

        # Load test set - handle the weights_only warning
        try:
            test_dataset = torch.load(TESTSET_PATH, weights_only=True)
        except:
            # If weights_only=True fails, fall back to original method
            print("Warning: Loading test dataset with weights_only=False")
            test_dataset = torch.load(TESTSET_PATH, weights_only=False)
        
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        for idx, (img, label) in enumerate(test_loader):
            if idx >= 5:
                break  # Limit to 5 images for visualization

            img = img.to(DEVICE)

            apply_gradcam(cnn_model, img, cnn_target_layer, label.item(), "CNN", idx)
            apply_gradcam(deit_model, img, deit_target_layer, label.item(), "DeiT", idx, is_vit=True)

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Verify that the model checkpoint matches the architecture you're trying to load")
        print("2. Check if the checkpoint contains the full model or just the state_dict")
        print("3. Ensure the number of classes in your model matches the checkpoint")