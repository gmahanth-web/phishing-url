---
title: Phishing URL Detector
emoji: 🎣
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Phishing URL Detector

A machine learning system that classifies URLs as **phishing** or **legitimate** using lexical (URL-structure-only) features and XGBoost — built to work in real time, without needing to visit or scrape the target page first.

**Live demo:** https://phishing-url-detector.onrender.com

*(Note: free-tier hosting spins down after inactivity — first load may take 30-50 seconds to wake up.)*

---

## Overview

This project started with a model trained on the [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) that hit 100% accuracy — a red flag, not a win. From there, the project became a methodical investigation into *why* the model was performing too well, and a fix for a real generalization gap discovered along the way.

## Key steps

1. **Identified and removed data leakage** — several features (e.g. `URLSimilarityIndex`, `TLDLegitimateProb`) were derived using information correlated with the label itself during dataset construction. Removed all leaky and post-visit scraping features to keep only genuinely real-time-available URL features.
2. **Validated with domain-grouped splits** — ensured URLs from the same domain never appeared in both train and test sets, preventing an easier form of leakage.
3. **Found a real-world generalization gap** — after achieving 99.7% accuracy on the benchmark test set, the model still misclassified simple, everyday domains like `google.com` and `amazon.com` as phishing with near-100% confidence.
4. **Root-caused it with SHAP** — used SHAP value analysis to identify that the training data underrepresented short, simple root-domain URLs, causing the model to associate low URL complexity with phishing.
5. **Fixed it with targeted data augmentation** — sampled ~4,000 real domains from the [Tranco top-sites list](https://tranco-list.eu/) across multiple popularity tiers, added them as legitimate examples, and retrained.
6. **Validated honestly** — held out part of the augmented data as a test set the model never trained on. Result: **99.6% accuracy on real-world domains never seen during training.**

## Tech stack

- Python, XGBoost, scikit-learn, pandas
- SHAP (model interpretability / debugging)
- Gradio (interactive demo interface)
- PhiUSIIL dataset + Tranco top-sites list (data augmentation)
- Deployed on Render

## Repository contents

| File | Purpose |
|---|---|
| `training_pipeline.py` | Full end-to-end training pipeline: data loading, leakage removal, feature extraction, Tranco augmentation, final model training |
| `app.py` | Gradio app for real-time URL prediction (used for the live demo) |
| `requirements.txt` | Python dependencies |
| `phishing_xgboost_final.pkl` | Trained final model |
| `tld_category_map.pkl` | Saved TLD category encoding, required for consistent inference |

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints in your terminal.

## Results

| Metric | Score |
|---|---|
| Benchmark test accuracy (PhiUSIIL) | 99.6% |
| Held-out real-world domain accuracy (Tranco, unseen) | 99.6% |

## Limitations

This is a **lexical-only** model — it evaluates URL structure alone and does not inspect page content. This is a deliberate design choice to keep predictions available at real time (before visiting a potentially malicious page), but it means the model would not catch a phishing site using a legitimate-looking URL with malicious page content.

## Author

Mahanthi Gautham Aditya
