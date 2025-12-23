# Augmented Reality with Planar Homographies

This project builds a complete **augmented reality system** using planar
homographies, feature matching, and RANSAC.

**Full technical details:**  
[Project Report (PDF)](./augmented_reality_homography_report.pdf)

---

## Pipeline Overview

1. FAST corner detection + BRIEF descriptors
2. Feature matching with Hamming distance
3. Homography estimation with normalization
4. RANSAC for robustness
5. Image and video augmentation

---

## Feature Matching

Automatically detected and matched keypoints using **FAST corner detection**
and **BRIEF descriptors**, with Hamming distance and ratio test to filter
reliable correspondences.

![Feature Matching](./assets/2.1.4.png)

Screenshots from report:

![Feature Matching](./assets/2.png)
![Feature Matching](./assets/3.png)
![Feature Matching](./assets/1.png)

More tests about different ratio and rotation
can be found in [Project Report (PDF)](./augmented_reality_homography_report.pdf)

---

## Robust Homography with RANSAC


Estimated a planar homography using **RANSAC** to reject outlier matches,
resulting in a robust transformation under noisy feature correspondences.


---

## Image-based AR Result

Warped and composited a target image onto a planar surface using the estimated
homography, achieving a realistic augmented reality overlay.

Original image:

![RANSAC Homography](./assets/ori.png)

Result:

![RANSAC Homography](./assets/500_2.png)
![RANSAC Homography](./assets/10.png)

---

## Video-based AR

Tracked the planar target frame-by-frame and overlaid a source video in real time, demonstrating a complete **feature-based AR video pipeline**.

Click to watch: 
![AR Video Frame](./assets/result_ar.avi)

Image example:

![AR Video Frame](./assets/real3.png)
![AR Video Frame](./assets/real2.png)

---

