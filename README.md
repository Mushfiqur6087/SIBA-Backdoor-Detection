# Seeing the Invisible: Grad-CAM-Driven Detection of Sparse and Imperceptible Backdoor Attacks

A novel explainable backdoor detection method that identifies Sparse and Invisible Backdoor Attacks (SIBA) in deep neural networks by comparing model attention patterns using Grad-CAM.

## Overview

This repository implements a detection framework for sparse and invisible backdoor attacks, which are particularly challenging because their triggers cannot be seen by humans. Instead of using Grad-CAM for direct trigger localization, we treat it as a behavioral signal and compare attention patterns between a clean reference model and a suspected backdoored model.

## Key Features

- **Grad-CAM-based model comparison** for detecting SIBA-style backdoor attacks
- **30-dimensional feature extraction** from heatmap differences covering statistical, spatial, magnitude, similarity, and gradient-based features
- **Multiple classifier support** including traditional ML (XGBoost, LightGBM, Random Forest, SVM) and deep learning approaches (2D CNN, Spatial Attention)
- **High detection accuracy** of 98.34% with ROC-AUC of 99.78% using the Spatial Attention classifier

## Method

### Detection Pipeline

1. **Input Processing**: Clean and triggered images are prepared from the test set
2. **Grad-CAM Extraction**: Generate heatmaps from both clean and backdoored models using the last convolutional layer
3. **Heatmap Difference Computation**: Calculate attention differences between models
4. **Feature Extraction**: Extract 30 features across 5 categories from the difference maps
5. **Classification**: Train classifiers to distinguish normal vs. backdoor behavior

### Feature Categories

| Category | Features |
|----------|----------|
| Statistical | Mean, Std Dev, Max, Min, Median, Percentiles, Range |
| Spatial | Entropy, Center of Mass, Spatial Variance, Gini Coefficient |
| Magnitude | L1/L2/L∞ Norms, Energy Concentration |
| Similarity | Pearson Correlation, Cosine Similarity, SSIM, KL Divergence |
| Gradient | Gradient Mean, Edge Density, Smoothness, Frequency Energy |

## Results

### Detection Performance (CIFAR-10)

| Method | Test Accuracy | Test F1 | Test AUC |
|--------|---------------|---------|----------|
| XGBoost | 96.05% | 96.05% | 99.18% |
| Random Forest | 95.61% | 95.61% | 98.99% |
| LightGBM | 96.22% | 96.22% | 99.25% |
| SVM (RBF) | 96.54% | 96.54% | 99.31% |
| Soft Voting Ensemble | 97.68% | 97.68% | 99.62% |
| 2D CNN | 97.12% | 97.12% | 99.45% |
| **Spatial Attention** | **98.34%** | **98.34%** | **99.78%** |

### Key Findings

- Gradient-based features (especially Gradient Mean and Smoothness) are most important for detection
- Only 7 features capture ~80% of total importance
- The method achieves high recall (98.43%), minimizing missed backdoor detections

## Dataset

Experiments are conducted on **CIFAR-10**:
- 50,000 training images
- 10,000 test images
- 10 object classes
- Image size: 32×32 RGB

## Requirements

- Python 3.x
- PyTorch
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- OpenCV (for Grad-CAM visualization)

## Usage

```python
# 1. Train clean and backdoored models
clean_model = train_model(clean_data)
backdoored_model = train_model(poisoned_data)  # Using SIBA attack

# 2. Generate Grad-CAM heatmaps
heatmap_clean = grad_cam(model, image, target_layer='layer4')

# 3. Compute heatmap differences
D = H_suspected - H_clean

# 4. Extract features
features = extract_features(D)  # 30-dimensional vector

# 5. Classify
prediction = classifier.predict(features)
# 0 = clean, 1 = backdoor detected
```

## Citation

```bibtex
@inproceedings{siba_detection_2025,
  title={Seeing the Invisible: Grad-CAM-Driven Detection of Sparse and Imperceptible Backdoor Attacks},
  booktitle={2025 International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN)},
  year={2025},
  address={Rangpur, Bangladesh}
}
```

## Limitations

- Currently evaluated only on CIFAR-10 dataset
- Requires access to a clean reference model trained on the same task
- White-box access to models is assumed for Grad-CAM extraction

## Future Work

- Extend evaluation to larger datasets (ImageNet, etc.)
- Test against other backdoor attack types
- Develop methods that don't require a clean reference model

## License

This project is for research purposes.

## Acknowledgments

This work addresses the challenge of detecting Sparse and Invisible Backdoor Attacks (SIBA) as proposed by Gao et al. in IEEE TIFS 2024.
