# Lung X-Ray Segmentation using Conditional GAN (Pix2Pix)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project implements a **Pix2Pix Conditional Generative Adversarial Network (cGAN)** for automated lung segmentation from chest X-ray images. The model generates binary segmentation masks that accurately identify lung regions, supporting image-guided medical procedures and disease detection.

### Key Highlights

- **Model Architecture**: Pix2Pix GAN with U-Net generator and PatchGAN discriminator
- **Performance Metrics**: 
  - Average IoU: **89.42%**
  - Average Dice Coefficient: **93.82%**
- **Dataset**: Paired chest X-ray images with binary masks from the US National Library of Medicine

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Training](#model-training)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

## Background

Medical image segmentation is an essential and challenging task in medical imaging. The purpose of lung image segmentation is to extract the region of interest and facilitate:

- Image-guided surgery
- Disease detection (e.g., pulmonary tuberculosis, COVID-19, pneumonia)
- Automated diagnosis assistance
- Treatment planning

Lung images present unique challenges:
- Complex anatomical structures
- Low image quality
- Gray level similarity between different tissues
- High variability in pathological conditions

Pix2Pix GAN has demonstrated outstanding performance in various image-to-image translation tasks, making it ideal for medical image segmentation applications.

## Dataset

### Chest X-Ray Dataset with Binary Masks

- **Source**: United States National Library of Medicine
- **Focus**: Pulmonary tuberculosis identification
- **Format**: Paired dataset with X-ray images and corresponding binary lung masks
- **Image Size**: 512 × 512 pixels (resized and processed to 256 × 256)
- **Split**: 80% training, 20% testing

### Dataset Structure

```
dataset/
├── img/           # Chest X-ray images
├── mask/          # Binary segmentation masks
└── annotations/   # Annotation details
```

## Architecture

### Pix2Pix GAN

Pix2Pix is a general-purpose framework for image-to-image translation that learns a mapping from input images to output images. Unlike traditional GANs, it uses conditional inputs (X-ray images) rather than random noise.

### Generator: U-Net Architecture

The generator uses a **U-Net** architecture with:
- **Encoder**: Downsampling blocks that capture context
- **Decoder**: Upsampling blocks that enable precise localization
- **Skip Connections**: Preserve spatial information across the network

```
Input (256×256×3) → Encoder → Bottleneck → Decoder → Output (256×256×1)
```

**Key Features**:
- Encoder blocks with Leaky ReLU activation
- Decoder blocks with ReLU activation and Dropout
- Skip connections between encoder and decoder layers
- Batch normalization for stable training

### Discriminator: PatchGAN

The discriminator classifies whether image pairs are real or fake:
- **Input**: Concatenation of source image and target/generated mask
- **Architecture**: Convolutional network that outputs a patch-based classification
- **Objective**: Distinguish real pairs (X-ray + ground truth) from fake pairs (X-ray + generated mask)

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/lung-segmentation-cgan.git
cd lung-segmentation-cgan

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
tensorflow>=2.8.0
numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=9.0.0
scikit-learn>=1.0.0
```

## Usage

### Training the Model

```python
# Load and preprocess dataset
train_dataset = load_dataset('path/to/train')
test_dataset = load_dataset('path/to/test')

# Build generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Train the model
train(train_dataset, epochs=200)
```

### Running Inference

```python
# Load trained model
generator = load_model('path/to/checkpoint')

# Generate segmentation mask
xray_image = load_image('path/to/xray.jpg')
predicted_mask = generator(xray_image)

# Visualize results
display_results(xray_image, predicted_mask, ground_truth)
```

### Using the Notebook

Open and run the Jupyter notebook:

```bash
jupyter notebook Image_Translation_CGANS_lung_segmentation.ipynb
```

## Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Average IoU** | 89.42% |
| **Average Dice Coefficient** | 93.82% |

### Visualization

The model generates high-quality segmentation masks that accurately delineate lung boundaries:

- **Input**: Chest X-ray image
- **Generated Output**: Binary segmentation mask
- **Ground Truth**: Manually annotated mask

Results demonstrate the effectiveness of Pix2Pix GAN in medical image segmentation tasks.

## Model Training

### Data Preprocessing Pipeline

1. **Resize**: Images resized to 286 × 286 pixels
2. **Random Crop**: Cropped to 256 × 256 pixels for training
3. **Random Flip**: Horizontal flipping for data augmentation
4. **Normalization**: Pixel values normalized to [-1, 1]

### Loss Functions

**Generator Loss**:
```
Generator Loss = GAN Loss + λ × L1 Loss
```
- **GAN Loss**: Binary cross-entropy between discriminator predictions and real labels
- **L1 Loss**: Mean absolute error between generated and ground truth masks
- **Lambda (λ)**: Weight parameter (typically 100)

**Discriminator Loss**:
```
Discriminator Loss = Real Loss + Generated Loss
```

### Training Configuration

- **Optimizer**: Adam (β₁ = 0.5, β₂ = 0.999)
- **Learning Rate**: 2e-4 with decay
- **Batch Size**: 1 (for memory efficiency)
- **Epochs**: 200
- **Hardware**: GPU-accelerated training

### Learning Rate Scheduling

Learning rate decay applied during training to improve convergence and stability.

## Performance Metrics

### Intersection over Union (IoU)

IoU measures the overlap between predicted and ground truth masks:

```
IoU = Area of Overlap / Area of Union
```

**Average IoU: 89.42%** - Indicates excellent segmentation accuracy.

### Dice Coefficient

Dice coefficient measures the similarity between predicted and ground truth:

```
Dice = 2 × |Predicted ∩ Ground Truth| / (|Predicted| + |Ground Truth|)
```

**Average Dice Score: 93.82%** - Demonstrates high-quality segmentation performance.

## Project Structure

```
lung-segmentation-cgan/
├── Image_Translation_CGANS_lung_segmentation.ipynb  # Main notebook
├── Lung_Image_Segmentation_-_Part_3.pptx           # Project presentation
├── README.md                                        # This file
├── requirements.txt                                 # Python dependencies
├── models/
│   ├── generator.py                                # Generator architecture
│   └── discriminator.py                            # Discriminator architecture
├── utils/
│   ├── data_loader.py                              # Dataset loading utilities
│   ├── preprocessing.py                            # Image preprocessing
│   └── metrics.py                                  # Evaluation metrics
├── checkpoints/                                    # Trained model checkpoints
├── logs/                                           # TensorBoard logs
└── results/                                        # Generated outputs
```

## Key Features

- ✅ **U-Net Generator**: Preserves spatial information through skip connections
- ✅ **PatchGAN Discriminator**: Classifies patches for better local detail
- ✅ **Data Augmentation**: Random jittering and flipping for robust training
- ✅ **Mixed Loss Function**: Combines adversarial and L1 losses
- ✅ **TensorBoard Integration**: Real-time training visualization
- ✅ **High Performance**: 89.42% IoU and 93.82% Dice coefficient

## Future Improvements

- [ ] Implement 3D segmentation for CT scans
- [ ] Add support for multi-class segmentation (different lung pathologies)
- [ ] Explore attention mechanisms for improved feature learning
- [ ] Deploy as a web service for clinical use
- [ ] Experiment with other GAN architectures (CycleGAN, StyleGAN)

## References

1. Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. *CVPR*.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
3. US National Library of Medicine - Chest X-ray Dataset

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset provided by the United States National Library of Medicine
- Pix2Pix implementation inspired by the original paper by Isola et al.
- TensorFlow and Keras communities for excellent documentation

## Contact

For questions or collaboration opportunities, please open an issue or contact the repository maintainer.

---

**Note**: This project is for research and educational purposes. Clinical deployment requires extensive validation and regulatory approval.
