# Multimodal Sequence Prediction Project (Baseline)

## 1. Project Overview

This project focuses on predicting the next element in a multimodal sequence.  
Given a sequence of images and corresponding text descriptions, the model learns to predict the next image-text pair.

---

## 2. Dataset

We use the StoryReasoning dataset (daniel3303/StoryReasoning).  
It contains sequences of images along with text descriptions that represent the story context.

---

## 3. Baseline Model

The baseline model consists of the following components:

- Visual Encoder (CNN-based feature extractor)
- Text Encoder (LSTM/Transformer)
- Multimodal Fusion module
- Sequence Prediction network (LSTM-based)
- Dual decoders for image and text generation

The model learns to predict the next image and text based on previous sequence information without any additional grounding or alignment improvements.

---

## 4. Current Status

Baseline model training has been completed successfully.

---

## 5. Experiment 1: Grounding Improvements

In this experiment, the grounding module was improved using:
- Re-Identification Loss (ReID)
- Frame-aware Grounding MSE Loss

These changes help improve alignment between image regions and text descriptions.

### Modified Code Snapshot

```python
loss_reid = F.mse_loss(z_r1, z_r2)

loss_ground_mse = F.mse_loss(z_r1, z_t_match)
```

### Result

The model showed improved grounding between visual and textual representations.

![Experiment 1 Output 1](results/exp1_grounding_1.png)

![Experiment 1 Output 2](results/exp1_grounding_2.png)

![Experiment 1 Output 3](results/exp1_grounding_3.png)
