# Neural Network for Visual Recognition

> **From-scratch neural networks + OCR pipeline + unsupervised representation learning**  
> This project builds core vision learning components **from first principles** (NumPy), then applies them to **character recognition and OCR**, and finally explores **learning character structure without labels** (AutoEncoder + PSNR).

**Full technical details:**  
[Project Report (PDF)](./neural_network_report.pdf)

---

## Key Components

- **Neural network from scratch (NumPy)**: weight init, forward propagation, backpropagation, mini-batch SGD  
- **Correctness verification**: **finite-difference gradient checking** to validate backprop gradients  
- **Character classification** on visual datasets (e.g., NIST36-style 36-way classification)  
- **OCR system (hybrid CV + DL)**: classical vision for detection + neural network for recognition  
- **Unsupervised learning (AutoEncoder)**: learns compact character representations **without labels**  
- **Reconstruction quality metric**: **PSNR** for quantitative evaluation  
- **PyTorch implementations** (FC/CNN/transfer learning) for engineering comparison and scaling

---

## OCR System (How “image → text” works)

> **Detect first, recognize second** — a practical OCR design used in real systems.

- **Denoise** (e.g., bilateral filtering): suppress background texture while preserving stroke edges  
- **Binarize** (grayscale + Otsu threshold): separate dark text from light background  
- **Morphology** (dilation): connect broken strokes so each character becomes one connected region  
- **Connected components**: treat each connected region as a character candidate and extract **bounding boxes**  
- **Filter small regions**: remove dust/noise by area thresholding  
- **Group into lines + sort by x**: form readable text order; optionally insert spaces by gap heuristics  
- **Classify each crop** with the trained neural network and **assemble the final string**

---

## Learning Character Structure Without Labels (AutoEncoder)

> Instead of predicting labels, the model learns by **reconstructing the input**.

- Train an **AutoEncoder** to map: `image → compressed latent → reconstructed image`  
- The **bottleneck latent space** forces the network to store only the most important information  
  (stroke layout, geometry, shared character structure), not pixel-level noise  
- Evaluate reconstruction using **PSNR (Peak Signal-to-Noise Ratio)**: higher PSNR ⇒ closer to the original  
- Interpretation: the network learns **what characters look like** (structure), even without knowing **what they are called** (labels)

---

## Training Dynamics and Performance Evaluation

- Loss/accuracy curves for supervised classification  
- Confusion matrix analysis for common character confusions (e.g., O/0, I/1)  
- Visualizing learned weights/features (e.g., first-layer weight maps)  
- AutoEncoder reconstruction examples + **PSNR** quantitative scores  

More details and results are in the report:  
[Project Report (PDF)](./neural_network_report.pdf)

---
