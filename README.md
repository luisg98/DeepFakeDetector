# 🧠 DeepFakeDetector — Hybrid Deep Learning for Deepfake Image Detection

**Author:** Luís Gonçalves  
**Institution:** IPCA – M2AI, Braga, Portugal  
**Email:** a18851@alunos.ipca.pt  

## 🧩 Overview

**DeepFakeDetector** is a comprehensive research and implementation project focused on **detecting AI-generated (deepfake) images** using hybrid deep learning techniques.  

It compares and combines two state-of-the-art paradigms:

- 🧱 **Convolutional Neural Networks (CNNs)** — *EfficientNet-B0*  
- 🔷 **Vision Transformers (ViTs)** — *DeiT-Tiny*  
- 🧩 **Hybrid Ensemble** — late fusion (soft-voting) combining CNN and ViT outputs  
- 🔍 **Interpretability** — *Grad-CAM* visualization for explainable AI  

The system demonstrates that combining CNNs’ local texture sensitivity with ViTs’ global contextual awareness results in more accurate, robust, and interpretable deepfake detection.

> 🏆 **Final performance (ensemble model):**
> - Accuracy: **94.23%**
> - Precision: **92.67%**
> - Recall: **96.06%**
> - F1-Score: **94.33%**


## 📁 Repository Structure

# 🧠 DeepFakeDetector — Hybrid Deep Learning for Deepfake Image Detection

**Author:** Luís Gonçalves  
**Institution:** IPCA – M2AI, Braga, Portugal  
**Email:** a18851@alunos.ipca.pt  


## 🧩 Overview

**DeepFakeDetector** is a comprehensive research and implementation project focused on **detecting AI-generated (deepfake) images** using hybrid deep learning techniques.  

It compares and combines two state-of-the-art paradigms:

- 🧱 **Convolutional Neural Networks (CNNs)** — *EfficientNet-B0*  
- 🔷 **Vision Transformers (ViTs)** — *DeiT-Tiny*  
- 🧩 **Hybrid Ensemble** — late fusion (soft-voting) combining CNN and ViT outputs  
- 🔍 **Interpretability** — *Grad-CAM* visualization for explainable AI  

The system demonstrates that combining CNNs’ local texture sensitivity with ViTs’ global contextual awareness results in more accurate, robust, and interpretable deepfake detection.

> 🏆 **Final performance (ensemble model):**
> - Accuracy: **94.23%**
> - Precision: **92.67%**
> - Recall: **96.06%**
> - F1-Score: **94.33%**


## 📁 Repository Structure

DeepFakeDetector/
│
├── 01_DataGathering.py # Downloads dataset automatically from Kaggle
├── 02_ExploratoryDataAnalysis.ipynb # Exploratory Data Analysis (EDA)
├── 03_PreprocessAndLoadDataset.py # Dataset cleaning, augmentation, and DataLoaders
│
├── 11_ModelTrainingCNN.py # Train EfficientNet-B0 (CNN)
├── 12_ModelTrainingVIT.py # Train DeiT-Tiny (Vision Transformer)
│
├── 21_EvaluateAndVisualizeCNN.ipynb # Evaluate and visualize CNN performance
├── 22_EvaluateAndVisualizeVIT.ipynb # Evaluate and visualize ViT performance
├── 22_EvaluateEnsembleAndComparisonCNNVIT.ipynb # Compare and ensemble CNN + ViT
│
├── 31_Grad-CAMInterpretability.py # Grad-CAM interpretability visualizations
├── 41_ClassifySingleImage.py # Single-image classification (CLI/GUI)
│
├── Hybrid_Deep_Learning_Approaches_for_Deepfake_Image_Detection_Using_CNNs__Vision_Transformers__and_Interpretability_Techniques-5.pdf
├── LICENSE
└── README.md


## 🧠 Methodology Summary

| Stage | Description |
|-------|--------------|
| **1. Dataset Preparation** | Kaggle dataset *AI-Generated Images vs Real Images* (≈60k samples). Cleaned, resized (224×224), normalized, and split 64/16/20 into train/val/test. Includes extensive data augmentation (flip, rotation, affine, color jitter, perspective). |
| **2. CNN Model** | *EfficientNet-B0* fine-tuned on the dataset. Final convolutional blocks unfrozen; trained with Adam (1e-4), StepLR, and early stopping. |
| **3. Vision Transformer (ViT)** | *DeiT-Tiny* pretrained on ImageNet. Same training setup as CNN for fair comparison. Captures long-range dependencies using self-attention. |
| **4. Ensemble Strategy** | Late-fusion soft voting: average predicted probabilities from CNN and ViT → select highest combined score. Improves recall and reduces false positives. |
| **5. Interpretability** | *Grad-CAM* heatmaps show attention regions. CNN focuses on local texture anomalies (e.g., eyes, forehead); ViT captures global consistency. |


## 🧬 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|-----------|------------|----------|-----------|------|
| **EfficientNet-B0 (CNN)** | 91.24% | 90.24% | 92.48% | 91.34% | 0.972 |
| **DeiT-Tiny (ViT)** | 93.31% | 91.93% | 94.96% | 93.42% | 0.983 |
| **Ensemble (CNN + ViT)** | 🏆 **94.23%** | **92.67%** | **96.06%** | **94.33%** | **0.98** |

Grad-CAM visualizations confirm complementary behavior:
- **CNN:** attention on local texture artifacts.  
- **ViT:** attention on global semantic structures.  
- **Ensemble:** integrates both for robust, interpretable predictions.


## ⚙️ Installation

### Requirements

- Python 3.8+
- GPU (recommended)
- Major dependencies:
  ```bash
  torch>=2.2
  torchvision>=0.17
  timm>=1.0.3
  pandas>=2.0
  numpy>=1.26
  Pillow>=10.0
  matplotlib>=3.8
  scikit-learn>=1.4
  pytorch-grad-cam>=1.5.0
  kagglehub>=0.2.6
  wandb>=0.17  # optional

Setup

# 1. Clone repository
git clone https://github.com/luisg98/DeepFakeDetector.git
cd DeepFakeDetector

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # on macOS/Linux
venv\Scripts\activate      # on Windows

# 3. Install dependencies
pip install -r requirements.txt

🚀 Usage
1. Download Dataset

python 01_DataGathering.py

(Requires Kaggle credentials)
2. Preprocess Data

python 03_PreprocessAndLoadDataset.py

3. Train Models

python 11_ModelTrainingCNN.py   # Train EfficientNet-B0
python 12_ModelTrainingVIT.py   # Train DeiT-Tiny

4. Evaluate & Compare

Use the Jupyter notebooks under 21_*/22_* for visualization, ROC curves, and confusion matrices.
5. Interpretability

python 31_Grad-CAMInterpretability.py

6. Single-Image Prediction

python 41_ClassifySingleImage.py --image path/to/image.jpg

Outputs predicted label (Real / Fake) and Grad-CAM overlay.
🧮 Ensemble Logic

# Pseudo-code
p_cnn = model_cnn(img)
p_vit = model_vit(img)
p_final = (p_cnn + p_vit) / 2
prediction = torch.argmax(p_final)

This simple soft-voting approach consistently reduced both false positives and false negatives without retraining.
🔍 Explainability Example
Model	Grad-CAM Visualization
CNN (EfficientNet-B0)	Focuses on eyes, forehead — local textures
ViT (DeiT-Tiny)	Captures global face structure and lighting inconsistencies
Ensemble	Integrates both — best interpretability and accuracy
🧾 Citation

If you use this work in your research, please cite:

    Luís Gonçalves (2025). Hybrid Deep Learning Approaches for Deepfake Image Detection Using CNNs, Vision Transformers, and Interpretability Techniques.
    IPCA, M2AI, Braga, Portugal.

📜 License

This project is licensed under the MIT License.
See the LICENSE

file for details.
🤝 Contributing

Contributions are welcome!

    Fork this repo

    Create a feature branch (git checkout -b feature-X)

    Commit your changes (git commit -m "Add feature X")

    Push to your fork (git push origin feature-X)

    Open a Pull Request

📬 Contact

For questions or collaborations:

📧 a18851@alunos.ipca.pt

💻 GitHub – luisg98
