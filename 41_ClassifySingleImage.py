import os
import torch
from PIL import Image
from tkinter import Tk, filedialog
from torchvision import transforms, models
from timm import create_model

# ==========================
# CONFIGURATION
# ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['fake', 'real']
CNN_MODEL_PATH = "checkpoints/CNN/model_final.pth"
DEIT_MODEL_PATH = "checkpoints/DeiT/deit_final.pth"

# ==========================
# IMAGE TRANSFORM
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ==========================
# LOAD MODELS
# ==========================
def load_cnn_model():
    """Load CNN model with automatic architecture detection"""
    try:
        # First, inspect the checkpoint to determine architecture
        checkpoint = torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Determine architecture based on keys
        keys = list(state_dict.keys())
        
        if any('features.' in key for key in keys):
            # EfficientNet architecture detected
            print("Loading EfficientNet model...")
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        else:
            # ResNet architecture
            print("Loading ResNet18 model...")
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
        
        # Load the state dict with security fix
        model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=True))
        return model.to(DEVICE).eval()
        
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        print("Trying alternative loading methods...")
        
        # Fallback: try loading as EfficientNet (most likely based on your error)
        try:
            print("Attempting to load as EfficientNet...")
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
            model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=True))
            return model.to(DEVICE).eval()
        except Exception as e2:
            print(f"EfficientNet loading also failed: {e2}")
            raise e

def load_deit_model():
    """Load DeiT model with error handling"""
    try:
        model = create_model("deit_tiny_patch16_224", pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(DEIT_MODEL_PATH, map_location=DEVICE, weights_only=True))
        return model.to(DEVICE).eval()
    except Exception as e:
        print(f"Error loading DeiT model: {e}")
        raise e

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_image(model, image_tensor):
    """Make prediction with confidence score"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()
            confidence = torch.softmax(outputs, dim=1)[0][pred_idx].item()
        return CLASS_NAMES[pred_idx], confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "error", 0.0

def validate_image_path(image_path):
    """Validate that the selected image exists and is readable"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"Error: Invalid image file: {e}")
        return False

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    print("Deepfake Detection - Image Classifier")
    print("=" * 40)
    
    # Select image
    root = Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
    )

    if not image_path:
        print("No image selected. Exiting...")
        exit()

    # Validate image
    if not validate_image_path(image_path):
        exit()

    print(f"Selected image: {os.path.basename(image_path)}")
    print("Loading models...")

    try:
        # Load models
        cnn_model = load_cnn_model()
        print("✓ CNN model loaded successfully")
        
        deit_model = load_deit_model()
        print("✓ DeiT model loaded successfully")

        # Load and prepare image
        print("Processing image...")
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Make predictions
        print("Making predictions...")
        cnn_label, cnn_conf = predict_image(cnn_model, image_tensor)
        deit_label, deit_conf = predict_image(deit_model, image_tensor)

        # Display results
        print("\n" + "=" * 40)
        print("PREDICTION RESULTS")
        print("=" * 40)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"CNN Prediction : {cnn_label.upper():<5} ({cnn_conf * 100:.2f}%)")
        print(f"DeiT Prediction: {deit_label.upper():<5} ({deit_conf * 100:.2f}%)")
        
        # Ensemble prediction (optional)
        if cnn_label == deit_label:
            avg_conf = (cnn_conf + deit_conf) / 2
            print(f"Consensus     : {cnn_label.upper():<5} ({avg_conf * 100:.2f}%)")
        else:
            print("Consensus     : MIXED (models disagree)")
        
        print("=" * 40)

    except Exception as e:
        print(f"Error during execution: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your model checkpoints exist and are valid")
        print("2. Check that you have the required dependencies installed")
        print("3. Ensure you have enough GPU/CPU memory available")