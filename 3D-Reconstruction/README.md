# 3D Reconstruction from Two Views

This project implements a **classical 3D reconstruction pipeline**
from two-view and multi-view images based on **multiple-view geometry**.

The system follows a standard **Structure-from-Motion (SfM)** framework,
recovering camera geometry and 3D structure from sparse 2D correspondences.
All major components are implemented from scratch using linear algebra
and non-linear optimization.

**Full technical details:**  
[Project Report (PDF)](./3d_reconstruction_report.pdf)

---

## Pipeline Overview

1. Fundamental matrix estimation (Eight-point & Seven-point)
2. Essential matrix computation using camera intrinsics
3. Camera pose recovery and linear triangulation
4. Epipolar correspondence for automatic matching
5. RANSAC for robust estimation under outliers
6. Bundle Adjustment for geometric refinement
7. Multi-view 3D reconstruction

---

## Fundamental Matrix & Epipolar Geometry

### Eight-Point Algorithm
![Eight Point Epipolar Lines](./assets/2.1.png)

- Point normalization for numerical stability  
- SVD-based least squares with rank-2 constraint  
- Geometric error refinement  

### Seven-Point Algorithm
![Seven Point Epipolar Lines](./assets/2.2.png)

- Minimal solver enforcing `det(F)=0`  
- Produces multiple valid solutions  
- Used inside RANSAC  

---

## Triangulation & Metric Reconstruction

Recovered 3D structure of the temple scene using calibrated cameras.

![Triangulation Point Cloud](./assets/4.2.png)
![Temple Reconstruction](./assets/4.2.3.png)
![Temple Reconstruction](./assets/4.2.4.png)

- Essential matrix computation from `F` and camera intrinsics  
- Camera pose recovery from Essential Matrix decomposition  
- Linear triangulation with reprojection error validation  

---

## Epipolar Correspondence

Automatic correspondence search constrained along epipolar lines.

![Epipolar Matching GUI](./assets/4.1.png)

- Reduced 2D search to 1D along epipolar lines  
- Gaussian-weighted patch matching (SSD)  
- Enables semi-dense correspondences  

---

## Robust Estimation with RANSAC

Comparison between naive estimation and RANSAC under noisy matches.

![RANSAC vs Eight Point](./assets/5.1.png)

- Seven-point minimal solver  
- Inlier selection using epipolar error  
- Robust Fundamental Matrix estimation  

---

## Bundle Adjustment

Joint optimization of camera pose and 3D points.

- Initial reprojection error: **~925**  
- After optimization: **~9**

![Bundle Adjustment](./assets/5.3.png)

- Rodrigues rotation parameterization  
- Non-linear optimization of structure and motion  
- Significant reprojection error reduction  

---

## Multi-view 3D Reconstruction

3D reconstruction using multiple synchronized camera views.

![Multi-view Car Reconstruction](./assets/6.1.1.png)
![Multi-view Car Reconstruction](./assets/6.2.png)

- Multi-view triangulation  
- Confidence-aware reconstruction  
- 3D structure and motion visualization  

---

## Summary

This project implements a complete **SfM-style 3D reconstruction system**
from first principles, covering epipolar geometry, triangulation,
robust estimation, and non-linear optimization.
