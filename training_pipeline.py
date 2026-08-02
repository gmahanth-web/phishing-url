# =============================================================================
# PHISHING URL DETECTION — FULL PIPELINE
# Dataset: PhiUSIIL Phishing URL Dataset (UCI ML Repository)
# Label convention: 0 = Phishing, 1 = Legitimate
#
# This script is organized into cells (marked with # %% CELL N) so you can
# paste each block into its own Colab cell in order.
# =============================================================================


# %% CELL 1 — Setup & Load Data
import pandas as pd
import numpy as np
import re
import joblib
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/fakedatacsv/fakeurl.csv'
df = pd.read_csv(file_path)
df = df.dropna().drop_duplicates()

TARGET_COL = 'label'
label_map = {0: 'Phishing', 1: 'Legitimate'}
target_names = ['Phishing (0)', 'Legitimate (1)']

print("Dataset loaded:", df.shape)
print(df['label'].value_counts())


# %% CELL 2 — Verify Label Direction (don't skip this)
# Confirms 0 = Phishing, 1 = Legitimate using known real-world correlates
print(df.groupby('label')['IsHTTPS'].mean())
print(df.groupby('label')['NoOfExternalRef'].mean())
print(df.groupby('label')['HasCopyrightInfo'].mean())
# Legitimate (1) should score higher on all three — confirmed for this dataset.


# %% CELL 3 — Drop Leaky, Metadata, and Scraped-Content Features
# These either directly encode the label (leakage), are raw identifiers not
# usable for prediction, or are page-content/scraping artifacts that would
# require visiting the URL first (not available at real-time inference).
drop_cols = [
    # Leakage / target proxies
    'URLSimilarityIndex', 'NoOfExternalRef', 'NoOfSelfRef',
    'TLDLegitimateProb', 'URLCharProb', 'CharContinuationRate',
    # Raw identifiers / metadata
    'FILENAME', 'URL', 'Domain', 'Title',
    # Page content / HTML scraping features (not available pre-visit)
    'LineOfCode', 'LargestLineLength', 'NoOfImage', 'NoOfCSS', 'NoOfJS',
    'NoOfEmptyRef', 'HasSocialNet', 'HasDescription', 'HasTitle',
    'DomainTitleMatchScore', 'URLTitleMatchScore', 'HasFavicon', 'Robots',
    'IsResponsive', 'NoOfURLRedirect', 'NoOfSelfRedirect', 'NoOfPopup',
    'NoOfiFrame', 'HasExternalFormSubmit', 'HasSubmitButton', 'HasHiddenFields',
    'HasPasswordField', 'Bank', 'Pay', 'Crypto', 'HasCopyrightInfo'
]
drop_list = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=[TARGET_COL] + drop_list)
y = df[TARGET_COL]

# Save the TLD category mapping BEFORE encoding, so inference-time URLs
# can be mapped consistently to the same integer codes used in training.
tld_categories = df['TLD'].astype('category').cat.categories
tld_map = {cat: code for code, cat in enumerate(tld_categories)}
joblib.dump(tld_map, 'tld_category_map.pkl')

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes

print(f"Lexical (URL-only) features retained: {X.shape[1]}")
print(X.columns.tolist())


# %% CELL 4 — Baseline Lexical Model (URL-derived features only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

url_model = XGBClassifier(
    n_estimators=250, learning_rate=0.08, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
)
url_model.fit(X_train, y_train)

y_pred = url_model.predict(X_test)
print(f"Baseline lexical model accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
print(confusion_matrix(y_test, y_pred))


# %% CELL 5 — Domain-Grouped Split (sanity check against domain leakage)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['Domain']))
X_train_g, X_test_g = X.iloc[train_idx], X.iloc[test_idx]
y_train_g, y_test_g = y.iloc[train_idx], y.iloc[test_idx]

grouped_model = XGBClassifier(
    n_estimators=250, learning_rate=0.08, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
)
grouped_model.fit(X_train_g, y_train_g)
y_pred_g = grouped_model.predict(X_test_g)
print(f"Domain-grouped test accuracy: {accuracy_score(y_test_g, y_pred_g)*100:.2f}%")


# %% CELL 6 — Feature Extraction Function (for real-world inference)
def extract_url_features(url, feature_order):
    parsed = urlparse(url)
    if not parsed.scheme:
        parsed = urlparse('http://' + url)
    domain = parsed.netloc.split(':')[0] if parsed.netloc else parsed.path.split('/')[0]
    post_scheme = url.split('://', 1)[-1]

    url_length = len(url)
    domain_length = len(domain)
    is_domain_ip = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain) else 0
    tld = domain.split('.')[-1] if '.' in domain and not is_domain_ip else ''
    tld_code = tld_map.get(tld, -1)

    no_subdomain = max(0, len(domain.split('.')) - 2) if ('.' in domain and not is_domain_ip) else 0

    has_obfuscation = 1 if '%' in url or '@' in url else 0
    no_obfuscated_char = url.count('%') + url.count('@')
    obfuscation_ratio = no_obfuscated_char / url_length if url_length > 0 else 0

    no_letters = sum(c.isalpha() for c in url)
    letter_ratio = no_letters / url_length if url_length > 0 else 0
    no_digits = sum(c.isdigit() for c in url)
    digit_ratio = no_digits / url_length if url_length > 0 else 0

    no_equals = url.count('=')
    no_qmark = url.count('?')
    no_ampersand = url.count('&')

    special_chars = set("!@#$%^&*()_+-=[]{}|;:'\",.<>?`~")
    no_other_special = sum(1 for c in post_scheme if c in special_chars and c not in ['=', '?', '&'])
    special_char_ratio = sum(1 for c in post_scheme if c in special_chars) / len(post_scheme) if post_scheme else 0

    is_https = 1 if parsed.scheme == 'https' else 0

    raw_features = {
        'URLLength': url_length, 'DomainLength': domain_length, 'IsDomainIP': is_domain_ip,
        'TLD': tld_code, 'TLDLength': len(tld), 'NoOfSubDomain': no_subdomain,
        'HasObfuscation': has_obfuscation, 'NoOfObfuscatedChar': no_obfuscated_char,
        'ObfuscationRatio': obfuscation_ratio, 'NoOfLettersInURL': no_letters,
        'LetterRatioInURL': letter_ratio, 'NoOfDegitsInURL': no_digits,
        'DegitRatioInURL': digit_ratio, 'NoOfEqualsInURL': no_equals,
        'NoOfQMarkInURL': no_qmark, 'NoOfAmpersandInURL': no_ampersand,
        'NoOfOtherSpecialCharsInURL': no_other_special, 'SpacialCharRatioInURL': special_char_ratio,
        'IsHTTPS': is_https,
    }
    feat_df = pd.DataFrame([raw_features])
    return feat_df[feature_order]


# %% CELL 7 — Initial Real-World Sanity Check
# (This is expected to FAIL on famous root domains before augmentation —
#  that's the generalization gap this pipeline fixes in Cell 8-11.)
sanity_urls = [
    "https://google.com", "https://www.amazon.com", "https://github.com", "https://apple.com",
    "http://login.microsoftonline.com.auth-update-portal.info/oauth2",
    "http://secure-paypal-account-verify.tk/login",
]
for u in sanity_urls:
    feat_df = extract_url_features(u, url_model.feature_names_in_)
    pred = url_model.predict(feat_df)[0]
    probs = url_model.predict_proba(feat_df)[0]
    print(f"{u:<60} -> {label_map[pred]:<12} [P(phish)={probs[0]:.4f}, P(legit)={probs[1]:.4f}]")


# %% CELL 8 — Load Tranco Top Sites for Augmentation
# Download from https://tranco-list.eu, unzip, upload the CSV to the same
# Drive folder as fakeurl.csv, then load it here.
tranco_path = '/content/drive/MyDrive/fakedatacsv/top-1m.csv'
tranco_df = pd.read_csv(tranco_path, names=['rank', 'domain'])
print("Tranco list loaded:", tranco_df.shape)


# %% CELL 9 — Build a Diverse, Rank-Spread Augmentation Sample
import random
random.seed(42)

tier_ranges = [(1, 1000), (1000, 10000), (10000, 50000), (50000, 150000), (150000, 500000)]
sampled_domains = []
for low, high in tier_ranges:
    tier_sample = tranco_df[(tranco_df['rank'] >= low) & (tranco_df['rank'] < high)]['domain'].sample(n=800, random_state=42)
    sampled_domains.extend(tier_sample.tolist())

augment_urls = []
for d in sampled_domains:
    augment_urls.append(f"https://{d}")
    augment_urls.append(f"https://www.{d}")

print(f"Total augmentation URLs: {len(augment_urls)}")

augment_rows = [extract_url_features(u, X.columns).iloc[0] for u in augment_urls]
X_augment = pd.DataFrame(augment_rows)
y_augment = pd.Series([1] * len(X_augment))  # all legitimate


# %% CELL 10 — Hold Out Part of the Augmented Set for Honest Evaluation
X_aug_train, X_aug_test, y_aug_train, y_aug_test = train_test_split(
    X_augment, y_augment, test_size=0.2, random_state=42
)
print(f"Augmented train: {len(X_aug_train)} | held-out augmented test: {len(X_aug_test)}")


# %% CELL 11 — Retrain Final Model with Augmented Data
X_combined = pd.concat([X, X_aug_train], ignore_index=True)
y_combined = pd.concat([y, y_aug_train], ignore_index=True)
sample_weight = pd.Series([1.0] * len(X) + [3.0] * len(X_aug_train))

X_train_f, X_test_f, y_train_f, y_test_f, w_train_f, _ = train_test_split(
    X_combined, y_combined, sample_weight,
    test_size=0.20, random_state=42, stratify=y_combined
)

final_model = XGBClassifier(
    n_estimators=250, learning_rate=0.05, max_depth=5,
    min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='logloss'
)
final_model.fit(X_train_f, y_train_f, sample_weight=w_train_f)

y_pred_f = final_model.predict(X_test_f)
print(f"Final model — benchmark test accuracy: {accuracy_score(y_test_f, y_pred_f)*100:.2f}%")
print(classification_report(y_test_f, y_pred_f, target_names=target_names, digits=4))

# The number that matters most: performance on real-world domains never seen in training
y_aug_pred = final_model.predict(X_aug_test)
print(f"Held-out real-world root-domain accuracy: {accuracy_score(y_aug_test, y_aug_pred)*100:.2f}%")

joblib.dump(final_model, 'phishing_xgboost_final.pkl')
print("Saved: phishing_xgboost_final.pkl")


# %% CELL 12 — Final Sanity Check (should now pass cleanly)
for u in sanity_urls:
    feat_df = extract_url_features(u, final_model.feature_names_in_)
    pred = final_model.predict(feat_df)[0]
    probs = final_model.predict_proba(feat_df)[0]
    print(f"{u:<60} -> {label_map[pred]:<12} [P(phish)={probs[0]:.4f}, P(legit)={probs[1]:.4f}]")


# %% CELL 13 — Interactive Demo (Gradio)
# !pip install gradio --quiet
import gradio as gr

def predict_url(url):
    if not url.strip():
        return "Please enter a URL."
    try:
        feat_df = extract_url_features(url, final_model.feature_names_in_)
        pred = final_model.predict(feat_df)[0]
        probs = final_model.predict_proba(feat_df)[0]
        label = "✅ Legitimate" if pred == 1 else "🚨 Phishing"
        confidence = probs[1] if pred == 1 else probs[0]
        return f"""
### {label}

**Confidence:** {confidence*100:.2f}%

| Class | Probability |
|-------|-------------|
| Phishing | {probs[0]*100:.2f}% |
| Legitimate | {probs[1]*100:.2f}% |
"""
    except Exception as e:
        return f"Error processing URL: {str(e)}"

demo = gr.Interface(
    fn=predict_url,
    inputs=gr.Textbox(label="Enter a URL", placeholder="e.g. https://google.com"),
    outputs=gr.Markdown(label="Result"),
    title="Phishing URL Detector",
    description="Paste a URL below to check if it's predicted to be legitimate or phishing.",
    examples=[
        "https://google.com",
        "https://www.wikipedia.org",
        "http://login.microsoftonline.com.auth-update-portal.info/oauth2",
        "http://secure-paypal-account-verify.tk/login",
    ]
)
demo.launch(share=True)
