# DeepFake Detector

Hybrid CNN + Vision Transformer model for deepfake image detection. 94% accuracy on the held-out test set. Explanations via Grad-CAM.

Compares CNN (ResNet/EfficientNet) against ViT architectures and combines both in a hybrid classifier. Data pipeline, training, and evaluation are scripted end-to-end for full reproducibility.

## Stack

PyTorch, timm, Grad-CAM. GPU required for training; inference runs on CPU.
