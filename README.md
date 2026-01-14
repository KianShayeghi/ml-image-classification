# Machine Learning - Image Classification

This repository contains a PyTorch-based implementation of an image classification pipeline using convolutional neural networks (CNNs). The project focuses on:

- Training a CNN to classify small RGB images into five categories.
- Evaluating model performance.
- Studying the impact of adversarial attacks on image classifiers.

The work is organised as a single Jupyter notebook that walks through the full process: from data loading to model training, evaluation, and robustness analysis.

---

## ⚠️ Hardware Requirements

**This project is intended to be run with GPU acceleration.**

Training the convolutional neural network and generating adversarial examples is computationally expensive on a CPU and may be impractically slow.

You should run this notebook using one of the following:

- **Google Colab with GPU enabled** (recommended)  
- A **local machine with a CUDA-enabled GPU**

If you try to run everything on CPU, expect:
- Very long training times  
- Slow adversarial attack generation  
- Poor usability for experimentation

---

## Dataset

The project uses the **Linnaeus 5 32×32** image dataset.

- Images are RGB, resized to **32 × 32** pixels.
- The dataset is divided into **five classes**:
  - berry  
  - bird  
  - dog  
  - flower  
  - other  

The dataset is downloaded externally and extracted into a local directory structure that the notebook expects before training begins.

---

## Project Structure

The notebook follows a clear experimental pipeline:

1. **Data preparation**
2. **Model design**
3. **Training and evaluation**
4. **Adversarial robustness analysis**

All steps are implemented and explained directly in the notebook.

---

## Model Overview

The core model is a **convolutional neural network (CNN)** implemented in PyTorch.

### Architecture (conceptual)
- Multiple **convolutional layers** to extract spatial features from images.
- **Pooling layers** to reduce dimensionality and improve translation invariance.
- **Fully connected layers** to map extracted features to class probabilities.
- Final output layer predicts one of the five image categories.

The notebook explains how convolutional layers progressively transform raw pixel data into higher-level representations that are suitable for classification.

---

## Training Procedure

- The dataset is split into training and test sets.
- Data is loaded using PyTorch’s `DataLoader`.
- The model is trained using:
  - A standard supervised learning setup.
  - Cross-entropy loss for multi-class classification.
  - Stochastic gradient-based optimisation.
- Training progress is monitored through:
  - Loss values.
  - Classification accuracy on the test set.

The notebook compares performance across different configurations to illustrate how architectural and training choices affect results.

---

## Adversarial Attacks

A key part of this project is the study of **adversarial robustness**.

### What is implemented
- **Fast Gradient Sign Method (FGSM)** is used to generate adversarial examples.
- Small, carefully chosen perturbations are added to images to intentionally mislead the classifier.
- The perturbation strength is controlled by a parameter `ε` (epsilon).

### What is analysed
- How increasing `ε` makes attacks stronger.
- How classification accuracy deteriorates as adversarial noise increases.
- The trade-off between:
  - Imperceptibility of perturbations.
  - Effectiveness of the attack.

The notebook demonstrates that even models with strong performance on clean data can be highly vulnerable to adversarial inputs.

---

## Key Results and Observations

- CNNs are effective at learning meaningful visual features even from small 32×32 images.
- Performance improves significantly compared to simpler or unstructured approaches.
- However, the trained model is **not robust** to adversarial perturbations:
  - Small, human-imperceptible changes to images can drastically reduce accuracy.
- The experiments highlight an important practical issue in modern machine learning:
  > High test accuracy does not imply real-world robustness.

---

## Importance of this project

This repository is not just about building a classifier. It demonstrates three important ideas in modern machine learning:

1. **Representation learning**  
   CNNs automatically extract hierarchical visual features from raw data.

2. **Model evaluation beyond accuracy**  
   A model can perform well on standard test sets but still fail under small, adversarial changes.

3. **Security and reliability of ML systems**  
   Adversarial attacks expose vulnerabilities that matter in real-world applications such as:
   - Computer vision in security systems.
   - Medical image analysis.
   - Autonomous systems.

---

## How to run

1. Upload the notebook to Colab.  
2. Go to **Runtime → Change runtime type**.  
3. Set **Hardware accelerator** to **GPU**.  
4. Run all cells.

---

## Dependencies

The notebook relies on standard machine-learning libraries, including:

- Python
- PyTorch (with CUDA support)
- NumPy
- Matplotlib

All data handling, training, and evaluation are done within the notebook environment.

---

## Summary

This project provides a complete, end-to-end example of:

- Training a convolutional neural network for image classification.
- Evaluating model performance in a controlled experimental setting.
- Stress-testing the model using adversarial attacks to understand its limitations.

It is intended both as a **technical implementation** and as a **conceptual demonstration** of why robustness is a crucial concern in modern deep learning systems.
