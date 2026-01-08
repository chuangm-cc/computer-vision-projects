# Spatial Pyramid Matching for Scene Classification

This project implements a classic **scene classification system** based on  **Bag of Visual Words (BoW)** and **Spatial Pyramid Matching (SPM)**, following  
the seminal work by **Lazebnik et al. (CVPR 2006)**.

The project builds a **complete traditional computer vision pipeline**, from
low-level feature extraction to mid-level image representation and final
scene classification, without relying on deep convolutional networks.

**Full technical details:**  
[Project Report (PDF)](./spatial_pyramid_matching_report.pdf)

---

## Pipeline Overview

1. **Multi-scale filter bank feature extraction** in Lab color space  
2. **K-means clustering** to build a visual word dictionary  
3. **Wordmap generation** via nearest visual word assignment  
4. **Spatial Pyramid Matching (SPM)** for spatially-aware feature encoding  
5. **Scene classification** using nearest neighbor and neural network models  

---

## Multi-scale Filter Bank Responses

A predefined filter bank including **Gaussian**, **Laplacian of Gaussian**, and
**directional Gaussian derivatives** is applied at multiple scales in the
**Lab color space**.

Each pixel is represented by concatenated multi-scale filter responses,
capturing local texture, edges, and orientation information.

![Filter Bank Responses](./assets/1.1.2.png)

---

## Visual Words Dictionary

Filter responses sampled from training images are clustered using **K-means**
to form a dictionary of **K visual words**, which discretize continuous local
features into a symbolic representation.

Larger dictionaries provide finer visual distinctions at the cost of increased
computational complexity.

---

## Wordmap Visualization

Each pixel is assigned to its nearest visual word, producing a **wordmap**
that reflects the spatial distribution of local appearance patterns.

Increasing the dictionary size results in finer structural segmentation.

**K = 10**

![Wordmap K10](./assets/1.3.png)

**K = 40**

![Wordmap K40](./assets/1.3.2.png)

---

## Recognition Performance

### Baseline: Nearest Neighbor + SPM

Using **SPM features** with **histogram intersection distance** and
**1-nearest-neighbor classification**, the baseline system achieves:

- Overall accuracy: **~50.5%**

| GT \ Pred | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|----------|----|----|----|----|----|----|----|----|
| **C0** | 31 | 1  | 6  | 0  | 2  | 0  | 6  | 4  |
| **C1** | 2  | 34 | 1  | 6  | 2  | 1  | 0  | 4  |
| **C2** | 2  | 3  | 17 | 0  | 2  | 7  | 0  | 19 |
| **C3** | 3  | 4  | 2  | 31 | 6  | 1  | 0  | 3  |
| **C4** | 3  | 2  | 5  | 10 | 15 | 5  | 7  | 3  |
| **C5** | 2  | 0  | 7  | 2  | 5  | 28 | 5  | 1  |
| **C6** | 8  | 1  | 2  | 0  | 5  | 7  | 22 | 5  |
| **C7** | 3  | 4  | 9  | 0  | 3  | 3  | 4  | 24 |

---

### Improved Model: Neural Network Classifier

SPM features are further used as input to a **neural network classifier**,
replacing the nearest-neighbor decision rule and enabling non-linear
classification.

![NN Confusion Matrix](./assets/steps.png)

Performance improvements are achieved by:
1. Increasing dictionary size (**K**) for finer visual representation  
2. Increasing sampled pixels (**α**) for better dictionary learning  
3. Increasing SPM layers (**L**) to capture richer spatial layouts  

---

### Further Improvement

A TensorFlow-based neural network trained on SPM features further improves
classification accuracy by learning non-linear decision boundaries.

- New accuracy: **0.6875**

| GT \ Pred | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|----------|----|----|----|----|----|----|----|----|
| **C0** | 40 | 0  | 1  | 1  | 1  | 2  | 3  | 2  |
| **C1** | 2  | 33 | 5  | 5  | 1  | 0  | 1  | 3  |
| **C2** | 2  | 2  | 32 | 0  | 1  | 1  | 8  | 4  |
| **C3** | 1  | 4  | 0  | 36 | 6  | 0  | 3  | 0  |
| **C4** | 1  | 2  | 1  | 11 | 30 | 2  | 1  | 2  |
| **C5** | 1  | 1  | 8  | 1  | 2  | 32 | 2  | 3  |
| **C6** | 2  | 0  | 2  | 0  | 3  | 2  | 39 | 2  |
| **C7** | 1  | 4  | 4  | 1  | 2  | 3  | 2  | 33 |

---
