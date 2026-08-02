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

An XGBoost-based classifier that predicts whether a URL is phishing or legitimate,
trained on the PhiUSIIL dataset and augmented with Tranco top-sites data to fix
generalization gaps on common real-world root domains.

Paste any URL into the box above to test it.
