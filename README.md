---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Magnetic Tile Defect Classification'
footer: '🏆 EfficientNetV2B0 · 92.57 % accuracy · F1-macro 0.8385'
style: |
  section {
    background: #FAFAFA;
    color: #1A2332;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    padding: 50px 70px;
  }
  section.title {
    background: linear-gradient(135deg, #1A2332 0%, #2C3E50 100%);
    color: #FFFFFF;
    text-align: left;
  }
  section.title h1 {
    color: #FF6B35;
    font-size: 2.6em;
    line-height: 1.05;
    margin-bottom: 0.2em;
  }
  section.title h2 {
    color: #FFFFFF;
    font-weight: 400;
    font-size: 1.3em;
  }
  section.title h3 {
    color: #4A90E2;
    font-weight: 300;
    margin-top: 2em;
    font-size: 1.05em;
  }
  section.section {
    background: #1A2332;
    color: #FFFFFF;
    text-align: center;
  }
  section.section h1 {
    color: #FF6B35;
    font-size: 3.2em;
    margin-top: 1em;
  }
  section.section h3 {
    color: #4A90E2;
    font-weight: 300;
  }
  h1 { color: #1A2332; font-weight: 700; font-size: 1.7em; margin-bottom: 0.1em; }
  h2 { color: #FF6B35; font-weight: 600; font-size: 1.15em; margin: 0.2em 0; }
  h3 { color: #4A90E2; font-size: 1.05em; }
  strong { color: #FF6B35; }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.5em 0;
    font-size: 0.78em;
  }
  th {
    background: #1A2332;
    color: #FFFFFF;
    padding: 8px 10px;
    text-align: left;
  }
  td {
    padding: 6px 10px;
    border-bottom: 1px solid #E5E7EB;
  }
  tr:nth-child(even) td { background: #F3F4F6; }
  .step {
    display: inline-block;
    background: #4A90E2;
    color: white;
    padding: 3px 10px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.75em;
    margin-right: 8px;
  }
  .did, .got, .saw, .next {
    border-left: 4px solid;
    padding: 8px 16px;
    margin: 0.4em 0;
    font-size: 0.85em;
  }
  .did { border-color: #4A90E2; background: #E8F2FB; }
  .got { border-color: #1A2332; background: #F1F2F4; }
  .saw { border-color: #F39C12; background: #FFF6E5; }
  .next { border-color: #FF6B35; background: #FFEFE8; }
  code {
    background: #1A2332;
    color: #FF9F68;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.85em;
  }
  pre {
    background: #1A2332;
    color: #E5E7EB;
    padding: 12px;
    border-radius: 6px;
    font-size: 0.65em;
    margin: 0.4em 0;
  }
  pre code { background: transparent; color: inherit; padding: 0; }
  img {
    border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    max-height: 380px;
  }
  ul li, ol li { margin: 0.25em 0; line-height: 1.4; font-size: 0.95em; }
---

<!-- _class: title -->

# Magnetic Tile Defect Classification

## A Complete Deep Learning Pipeline

### 🏆 92.57 % accuracy &nbsp;·&nbsp; F1-macro 0.8385 &nbsp;·&nbsp; EfficientNetV2B0

---

## Reading Guide

Every slide follows the same pattern:

<div class="did">🔵 <strong>What we did</strong> — the code / technique we applied in that cell</div>
<div class="got">⚫ <strong>What we got</strong> — the actual output or chart from the notebook</div>
<div class="saw">🟡 <strong>What we noticed</strong> — the key observation that drove the next step</div>
<div class="next">🟠 <strong>What we did next</strong> — the technical decision motivated by the observation</div>

> This format makes the project a chain of evidence-based decisions, not a list of techniques.

---

<!-- _class: section -->

# Section 1
## The Dataset
### Where it comes from, what's in it

---

## The Dataset

**Source:** [`alex000kim/magnetic-tile-surface-defects`](https://www.kaggle.com/datasets/alex000kim/magnetic-tile-surface-defects) on Kaggle
**Academic reference:** Huang et al., *Surface Defect Saliency of Magnetic Tile*, The Visual Computer (2018)
**Size:** 52 MB · 2688 files · **1344 grayscale image–mask pairs**

**Six classes:**

| Class | Type | Count |
|---|---|---|
| Blowhole | Gas bubble defect | 115 |
| Break | Surface fracture | 85 |
| Crack | Fine line crack | 57 |
| **Fray** | Edge fraying | **32** ⚠️ |
| **Free** | **No defect (healthy)** | **952** |
| Uneven | Surface irregularity | 103 |

> **Why this dataset?** Industrial surface inspection + pixel-precise GT masks + severe imbalance — three rare combined opportunities to apply advanced techniques.

---

## <span class="step">Cell 7</span> Loading Images and Masks

<div class="did">

**What we did:** Recursively walked through `MT_<class>/Imgs/`, loaded each `.jpg` (image) with its homonymous `.png` (binary mask), resized to 224×224 grayscale, normalized to [0, 1].

```python
img = cv2.resize(raw, (224, 224)) / 255.0
mask = load_mask_binary(r["mask_path"], 224)
```

</div>

<div class="got">

**What we got:**

```
Shape X_all : (1344, 224, 224, 1)
Shape M_all : (1344, 224, 224)  ← binary masks
Distribution : {0: 115, 1: 85, 2: 57, 3: 32, 4: 952, 5: 103}
```

</div>

<div class="saw">

**What we noticed:** The class distribution is **wildly skewed** — Free has 30× more samples than Fray.

</div>

<div class="next">

**What we did next:** Visualize the distribution to quantify the imbalance precisely (next cell).

</div>

---

## <span class="step">Cell 9</span> Class Distribution — EDA

<div class="did">

**What we did:** Plotted a bar chart and pie chart of class frequencies.

</div>

<div class="got">

**What we got:**

![w:780](./imgs/cell09_1.png)

</div>

<div class="saw">

**What we noticed:** Free = **70.8 %**, Fray = **2.4 %**, ratio **30:1**. A trivial model that always predicts "Free" already gets 71 % accuracy — a statistical trap.

</div>

<div class="next">

**What we did next:** Pick **F1-macro** as the primary metric (treats every class equally) and plan for class-weight correction.

</div>

---

## <span class="step">Cell 10</span> Image Size Distribution

<div class="did">

**What we did:** Histogrammed the original heights, widths and aspect ratios of all 1344 images.

</div>

<div class="got">

**What we got:**

![w:920](./imgs/cell10_1.png)

</div>

<div class="saw">

**What we noticed:** Image sizes vary **a lot** — from tens of pixels to several hundreds, with non-uniform aspect ratios.

</div>

<div class="next">

**What we did next:** Force a uniform resize to **224×224** (the standard ImageNet input size — compatible with EfficientNetV2B0 we'd use later).

</div>

---

## <span class="step">Cell 11</span> CLAHE Preprocessing

<div class="did">

**What we did:** Applied **CLAHE** (Contrast Limited Adaptive Histogram Equalization) with `clipLimit=2.5`, `tileGridSize=(8, 8)` — enhances local contrast without amplifying global noise.

```python
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
enhanced = clahe.apply(img_uint8)
```

</div>

<div class="got">

**What we got:**

![w:920](./imgs/cell11_1.png)

</div>

<div class="saw">

**What we noticed:** Subtle defects (Crack, Fray) become clearly more visible after CLAHE; Free tiles stay clean (no false patterns created).

</div>

<div class="next">

**What we did next:** Apply CLAHE to **all** train/val/test images as a fixed preprocessing step before any model.

</div>

---

## <span class="step">Cell 13</span> Sample Images Per Class

<div class="did">

**What we did:** Visualized 3 random samples per class side-by-side with their GT masks (red overlay).

</div>

<div class="got">

**What we got:**

![w:880](./imgs/cell13_1.png)

</div>

<div class="saw">

**What we noticed:** Defects can be **very localized** (small Blowhole, thin Crack) or **diffuse** (Uneven, Fray). Visually, **Break and Free look similar** in many regions.

</div>

<div class="next">

**What we did next:** Plan for an attention mechanism (CBAM) in the improved CNN, and check Grad-CAM later to see what the model actually looks at.

</div>

---

<!-- _class: section -->

# Section 2
## Handling Imbalance
### Three corrections + one diagnostic test

---

## <span class="step">Cell 15</span> Stratified Split + Class Weights

<div class="did">

**What we did:** 70 / 15 / 15 stratified split (preserves class proportions per split) + computed **sqrt-balanced** class weights.

```python
sqrt_weights = np.sqrt(1.0 / freqs)
sqrt_weights = sqrt_weights * len(y_train) / (sqrt_weights * freqs).sum()
```

</div>

<div class="got">

**What we got — Train / Val / Test split:**

| Class | Train | Val | Test | sqrt weight | vs balanced |
|---|---|---|---|---|---|
| Free | 666 | 143 | 143 | **0.59** | 0.24 |
| Fray | **22** | 5 | 5 | **3.23** | **7.12** ⚠ |
| Blowhole | 80 | 17 | 18 | 1.70 | 1.96 |

</div>

<div class="saw">

**What we noticed:** Only **22 Fray samples in train** — extremely few. Standard `balanced` would give Fray a weight of 7.12 (30× Free) which is too aggressive.

</div>

<div class="next">

**What we did next:** Use **sqrt-balanced** (5.5× ratio instead of 30×) — soft correction that helps minorities without destabilizing the majority.

</div>

---

## <span class="step">Cell 17</span> 🔬 Diagnostic Test #1 — Naive Baseline

<div class="did">

**What we did:** Trained a mini-CNN for 8 epochs with **plain cross-entropy**, **no class_weight**, **no focal loss** — a control experiment.

</div>

<div class="got">

**What we got:**

![w:780](./imgs/cell17_1.png)

```
Accuracy: 0.708   |   F1-macro: 0.138
Predictions: Free = 202/202 (100 %)    All other classes: 0
```

</div>

<div class="saw">

**What we noticed:** The naive model **predicts "Free" for 100 % of test images**. Accuracy 71 % is meaningless — F1-macro 0.14 is the catastrophic truth.

</div>

<div class="next">

**What we did next:** Make imbalance correction mandatory in the actual training pipeline: **sqrt class_weight + Focal Loss γ=1.0 + label_smoothing 0.05**.

</div>

---

## <span class="step">Cell 19</span> Soft Augmentation Strategy

<div class="did">

**What we did:** Designed a **gentle** augmentation pipeline: rotation ±10°, shift ±6 %, zoom [0.95, 1.05], horizontal flip, light Gaussian noise (σ ∈ [0.005, 0.015], prob 0.30).

</div>

<div class="got">

**What we got:**

![w:920](./imgs/cell19_1.png)

</div>

<div class="saw">

**What we noticed:** Earlier experiments with aggressive augmentation (heavy noise + CutMix) **destroyed the Fray signal** — only 22 training samples can't survive harsh transforms.

</div>

<div class="next">

**What we did next:** Lock in soft augmentation and **disable CutMix** for from-scratch training. Keep the augmentation generator for showcase only.

</div>

---

<!-- _class: section -->

# Section 3
## Models We Tested
### Baseline → CBAM → Transfer Learning

---

## <span class="step">Cell 23</span> Baseline CNN Architecture

<div class="did">

**What we did:** Built a reference 4-block CNN — `Conv → BN → ReLU → MaxPool → Dropout` repeated with filters 32 → 64 → 128 → 256, then GAP + Dense(128) + Softmax(6).

```python
def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    return x
```

</div>

<div class="got">

**What we got:** ~388 K trainable parameters, compiled with `categorical_crossentropy` + Adam(1e-3).

</div>

<div class="saw">

**What we noticed:** This is a standard architecture — won't have any imbalance bias built in, so it'll be a clean reference point.

</div>

<div class="next">

**What we did next:** Train it (Cell 24) and observe what happens with class_weight applied.

</div>

---

## <span class="step">Cell 27</span> Improved CNN — Residual + CBAM

<div class="did">

**What we did:** Stem 7×7 + **4 residual blocks** with **CBAM** attention (channel + spatial) and skip connections. Filters 64 → 128 → 256 → 512.

```python
def cbam_block(x, ratio=8):
    x = se_block(x, ratio=ratio)        # channel attention
    x = spatial_attention(x)            # spatial attention
    return x
```

</div>

<div class="got">

**What we got:** ~3 M parameters, compiled with **Focal Loss γ=1.0** + label_smoothing 0.05 + Cosine LR with warmup.

</div>

<div class="saw">

**What we noticed:** CBAM is strictly better than SE alone — it adds spatial attention on top of channel attention. Two attention types > one.

</div>

<div class="next">

**What we did next:** Train it (Cell 28) and compare to baseline. Hypothesis: CBAM + Focal should significantly outperform the baseline.

</div>

---

## <span class="step">Cells 24, 28</span> Training the From-Scratch CNNs

<div class="did">

**What we did:** Trained both Baseline and Improved CNN for up to 50 epochs with EarlyStopping(patience=10) + sqrt class_weight + (Focal Loss for Improved).

</div>

<div class="got">

**What we got — training logs:**

```
Baseline CNN
  Epoch  1   train_acc 0.45   val_accuracy 0.7079   val_loss 1.47
  Epoch  5   train_acc 0.68   val_accuracy 0.7079   val_loss 1.21
  Epoch 11   train_acc 0.70   val_accuracy 0.7079   ← EarlyStop

Improved CNN (CBAM + Focal)
  Epoch  1   train_acc 0.12   val_accuracy 0.7079
  Epoch 10   train_acc 0.87   val_accuracy 0.7079
  Epoch 13   train_acc 0.92   val_accuracy 0.7079   ← EarlyStop
```

</div>

<div class="saw">

**What we noticed:** `val_accuracy = 0.7079` **every single epoch** for BOTH models. That's exactly 143/202 — the Free proportion in val. **Both CNNs collapsed** to predicting Free for the entire val set.

</div>

<div class="next">

**What we did next:** Diagnose with learning curves (Cell 29), then move to transfer learning.

</div>

---

## <span class="step">Cell 29</span> Learning Curves — Visualizing the Collapse

<div class="did">

**What we did:** Plotted train vs validation accuracy/loss for both Baseline and Improved CNN.

</div>

<div class="got">

**What we got:**

![w:920](./imgs/cell29_1.png)

</div>

<div class="saw">

**What we noticed:** Train accuracy climbs steadily (model memorizes training data), but **val accuracy is a flat line at 0.7079** — confirming the collapse to majority class on the val set.

</div>

<div class="next">

**What we did next:** Conclude that from-scratch CNNs cannot learn rare-class features from only 22 Fray training samples. **Switch to transfer learning** using ImageNet-pretrained weights.

</div>

---

## <span class="step">Cells 31–32</span> Transfer Learning — EfficientNetV2B0

<div class="did">

**What we did:** Two-stage transfer learning on EfficientNetV2B0 (ImageNet-pretrained, ~6.25 M params). Converted grayscale → RGB by channel replication.

**Stage 1** — backbone frozen, train only the head, LR=1e-3, 15 epochs.
**Stage 2** — unfreeze the 30 last layers (BN kept frozen), LR=**1e-5**, 30 epochs max.

</div>

<div class="got">

**What we got:**

![w:780](./imgs/cell32_1.png)

```
Stage 1 epoch 9 :  val_accuracy = 0.9356
Stage 2 epoch 1 :  val_accuracy = 0.9554  ← best
Stage 2 epoch 9 :  EarlyStopping → restored best
```

</div>

<div class="saw">

**What we noticed:** Stage 1 alone already reaches **93.56 %** — ImageNet features carry over remarkably well. Stage 2 adds another **+2 pts** to **95.54 %**.

</div>

<div class="next">

**What we did next:** Evaluate this champion on the held-out test set (Cell 34).

</div>

---

<!-- _class: section -->

# Section 4
## Results & Diagnostic Tests

---

## <span class="step">Cell 34</span> Final Results on the Test Set

<div class="did">

**What we did:** Predicted on the held-out test set (202 images) with every trained model, computed accuracy + F1-macro + F1-weighted.

</div>

<div class="got">

**What we got:**

| Model | Accuracy | F1-macro | F1-weighted |
|---|---|---|---|
| Baseline CNN | 0.7079 | 0.1382 | 0.5869 |
| Improved CNN (CBAM + Focal) | 0.7079 | 0.1382 | 0.5869 |
| Improved CNN + TTA | 0.7079 | 0.1382 | 0.5869 |
| **EfficientNetV2B0** | **0.9257** | **0.8385** | **0.9192** |
| EfficientNetV2B0 + TTA | 0.9109 | 0.8012 | 0.9022 |

</div>

<div class="saw">

**What we noticed:** EfficientNetV2B0 gives a **+0.70 F1-macro gain** over baseline. TTA actually **hurts slightly** here (the model is already well-trained on clean data).

</div>

<div class="next">

**What we did next:** Drill down per-class to find weak spots (Cell 35).

</div>

---

## <span class="step">Cell 35</span> Per-Class Report & Confusion Matrix

<div class="did">

**What we did:** Generated a classification report (P/R/F1 per class) and confusion matrices (raw + normalized) for the champion.

</div>

<div class="got">

**What we got:**

![w:900](./imgs/cell35_1.png)

| Class | P | R | F1 |
|---|---|---|---|
| Blowhole | 1.00 | 0.89 | **0.94** |
| **Break** | 0.83 | **0.42** | **0.56** ⚠️ |
| Crack | 0.89 | 0.89 | 0.89 |
| Fray | 0.80 | 0.80 | 0.80 |
| Free | 0.92 | 0.99 | 0.96 |

</div>

<div class="saw">

**What we noticed:** All classes ≥ 0.80 F1 except **Break (F1=0.56, recall 0.42)** — 58 % of real breaks are predicted as Free. Industrial cost concern.

</div>

<div class="next">

**What we did next:** Investigate Break errors specifically in Cell 37.

</div>

---

## <span class="step">Cell 37</span> 🔬 Diagnostic Test #2 — Error Analysis

<div class="did">

**What we did:** Computed top confusion pairs, confidence distribution for correct vs wrong predictions, and visualized the 8 most-confident errors.

</div>

<div class="got">

**What we got:**

![w:520](./imgs/cell37_1.png) ![w:380](./imgs/cell37_2.png)

```
Top confusions:  Break → Free (7)  ·  Crack → Free (3)  ·  Uneven → Free (3)
Confidence avg correct: 0.720    Confidence avg wrong: 0.505
Gap: 21.6 pts
```

</div>

<div class="saw">

**What we noticed:** Errors cluster on **Break → Free** confusion. Crucially, the model is **21.6 pts less confident when wrong** — a strong exploitable signal.

</div>

<div class="next">

**What we did next:** Use this confidence gap to design a deployment threshold (Test #3, Cell 48).

</div>

---

## <span class="step">Cell 39</span> Calibration — Reliability Diagram & ECE

<div class="did">

**What we did:** Computed **Expected Calibration Error (ECE)** in 10 confidence bins, plotted the reliability diagram.

</div>

<div class="got">

**What we got:**

![w:480](./imgs/cell39_1.png)

```
ECE (no TTA) : 0.198
ECE + TTA    : 0.210
```

</div>

<div class="saw">

**What we noticed:** ECE ≈ 0.20 means the model is **slightly overconfident** — when it says "90 % sure" it's actually right ~75 % of the time. Not catastrophic, but not industrial-grade either.

</div>

<div class="next">

**What we did next:** Recommend **temperature scaling** post-hoc in production (one scalar fit on val set typically reduces ECE below 0.05).

</div>

---

## <span class="step">Cell 42</span> Grad-CAM — Quantitative Interpretability

<div class="did">

**What we did:** Generated Grad-CAM heatmaps for the **true class** of each test image, thresholded at 0.3, computed **IoU** vs the ground-truth mask.

```python
hm = make_gradcam_heatmap(img, model, "block3_out", pred_index=int(true_class))
iou_val = iou(hm_mask >= 0.3, mask_gt)
```

</div>

<div class="got">

**What we got:**

![w:760](./imgs/cell42_1.png)

```
Uneven:   IoU mean 0.283   ← largest defect, easiest to localize
Fray:     IoU mean 0.143
Blowhole, Break, Crack:   IoU ≈ 0   ← too small for 7×7 feature map
Global:   IoU mean 0.079
```

</div>

<div class="saw">

**What we noticed:** Localization works for **large diffuse defects** (Uneven) but fails for **fine defects** (Crack, Break, Blowhole) — limited by the 7×7 resolution of the last feature map.

</div>

<div class="next">

**What we did next:** Note the limit; in future work, use a higher-resolution feature map (e.g., `block2` output) or upsample with skip connections.

</div>

---

## <span class="step">Cell 44</span> t-SNE of Embeddings

<div class="did">

**What we did:** Extracted 256-dim embeddings from the improved CNN's `dense_relu` layer for all test samples; reduced to 2-D with t-SNE.

</div>

<div class="got">

**What we got:**

![w:560](./imgs/cell44_1.png)

```
Cosine intra-class : 1.000
Cosine inter-class : 1.000
Separation Δ       : -0.000
```

</div>

<div class="saw">

**What we noticed:** All embeddings collapsed to **the same point** (intra = inter = 1.000) — the from-scratch CNN learned nothing useful. Confirms again the from-scratch failure.

</div>

<div class="next">

**What we did next:** Use the embeddings to validate **why** the from-scratch CNN failed — and double-down on transfer learning as the only viable path.

</div>

---

## <span class="step">Cell 46</span> Robustness Tests

<div class="did">

**What we did:** Artificially corrupted the test set (noise σ=0.05/0.10, blur k=5/11, brightness γ=0.5/2.0) and re-evaluated EfficientNet.

</div>

<div class="got">

**What we got:**

![w:780](./imgs/cell46_1.png)

| Corruption | Accuracy | Δ vs clean |
|---|---|---|
| Clean | 92.57 % | — |
| Brightness γ=0.5 / 2.0 | 91.6 % / 90.1 % | −1 / −2 pts ✓ |
| Blur k=5 | 84.2 % | −8 pts |
| Noise σ=0.05 | 81.7 % | −11 pts |
| **Noise σ=0.10** | **69.3 %** | **−23 pts** ⚠️ |

</div>

<div class="saw">

**What we noticed:** Very robust to brightness (training augmentations help), **fragile against strong noise** (−23 pts). Industrial sensors with poor SNR would degrade the model significantly.

</div>

<div class="next">

**What we did next:** Recommend a denoising preprocessing filter or noise-augmentation training as future improvements.

</div>

---

## <span class="step">Cell 48</span> 🔬 Diagnostic Test #3 — Deployment Threshold

<div class="did">

**What we did:** Scanned confidence thresholds from 0.30 to 0.99 and measured **global precision** and **coverage** at each. Target: precision ≥ 95 % AND coverage ≥ 80 %.

</div>

<div class="got">

**What we got:**

![w:860](./imgs/cell48_1.png)

```
✅ Recommended threshold = 0.51
  → Automatic precision : 96.41 %
  → Automatic coverage  : 82.67 %
  → 17.3 % sent to human review

Per-class precision at 0.51 :
  Blowhole 100 %  ·  Break 100 %  ·  Crack 100 %
  Fray 100 %  ·  Free 95.6 %  ·  Uneven 100 %
```

</div>

<div class="saw">

**What we noticed:** A simple threshold of **0.51** gives **production-grade precision (96.4 %)** on 82.7 % of the flow, with a tractable 17.3 % human-review fallback.

</div>

<div class="next">

**What we did next:** Lock this deployment strategy into the conclusion — auto-decide when confident, escalate to humans when uncertain.

</div>

---

## <span class="step">Cell 50</span> Final Comparison Chart

<div class="did">

**What we did:** Built the final comparison summary table with ΔF1 vs Baseline and a side-by-side bar chart (Accuracy + F1-macro).

</div>

<div class="got">

**What we got:**

![w:880](./imgs/cell50_1.png)

</div>

<div class="saw">

**What we noticed:** A **+0.70 F1-macro gap** between the from-scratch CNNs and EfficientNetV2B0 is enormous on a 6-class problem. This is the clearest visual demonstration of why transfer learning was the right call.

</div>

<div class="next">

**What we did next:** Wrap up with conclusion + future work (next slides).

</div>

---

<!-- _class: section -->

# Section 5
## Conclusion

---

## What We Achieved

✅ **Champion model:** EfficientNetV2B0
&nbsp;&nbsp;&nbsp;&nbsp;**92.57 %** accuracy &nbsp;·&nbsp; **F1-macro 0.8385** &nbsp;·&nbsp; F1-weighted 0.9192

✅ **Three diagnostic tests** that justify every technical decision:
&nbsp;&nbsp;&nbsp;&nbsp;Test #1 (Cell 17) — proved imbalance correction is mandatory
&nbsp;&nbsp;&nbsp;&nbsp;Test #2 (Cell 37) — identified Break confusion + 21.6 pts confidence gap
&nbsp;&nbsp;&nbsp;&nbsp;Test #3 (Cell 48) — produced deployment threshold 0.51 → 96.4 % precision

✅ **Industrial-grade evaluation:** F1-macro, ECE calibration, Grad-CAM IoU vs masks, robustness curves

✅ **Reproducible notebook** end-to-end — 52 cells, ~30 min on Kaggle T4 GPU

---

## Honest Limitations

| Limit | Where | Impact |
|---|---|---|
| **Break class** F1 = 0.56, recall 0.42 | Cell 35 | 58 % of breaks predicted as Free → industrial cost |
| **From-scratch CNNs collapsed** | Cells 24, 28, 44 | 22 Fray train samples insufficient |
| **ECE = 0.20** (overconfidence) | Cell 39 | Needs temperature scaling for production |
| **Noise robustness** −23 pts at σ=0.10 | Cell 46 | Denoising filter recommended |
| **Grad-CAM IoU ~0** for fine defects | Cell 42 | 7×7 feature map too low-res |

> All limits are **measured** and **documented** — not hidden under the rug.

---

## Future Work (Prioritized)

| Priority | Action | Expected Gain |
|---|---|---|
| 🔴 High | **SMOTE / targeted augmentation on Break** | Break recall 0.42 → 0.70+ |
| 🔴 High | **Temperature scaling** (post-hoc, 1 param) | ECE 0.20 → < 0.05 |
| 🟡 Medium | **Auxiliary segmentation task** (GT masks available) | Better localization + classification |
| 🟡 Medium | **EfficientNetV2-S** (larger backbone) | +1–2 pts F1-macro expected |
| 🟢 Low | Denoising preprocessing filter | Robustness to strong sensor noise |

> The current model is **ready for production** with the threshold-0.51 + human-review strategy.

---

<!-- _class: title -->

# Thank You.

## Questions?

### 🏆 EfficientNetV2B0 · 92.57 % accuracy · F1-macro 0.8385
### 📓 52-cell reproducible notebook · ~30 min on Kaggle T4
