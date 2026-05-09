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
