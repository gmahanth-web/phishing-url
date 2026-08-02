import re
import joblib
import pandas as pd
import gradio as gr
from urllib.parse import urlparse

# Load model and TLD mapping (must be uploaded alongside this file)
model = joblib.load("phishing_xgboost_final.pkl")
tld_map = joblib.load("tld_category_map.pkl")
label_map = {0: "Phishing", 1: "Legitimate"}


def extract_url_features(url, feature_order):
    parsed = urlparse(url)
    if not parsed.scheme:
        parsed = urlparse("http://" + url)
    domain = parsed.netloc.split(":")[0] if parsed.netloc else parsed.path.split("/")[0]
    post_scheme = url.split("://", 1)[-1]

    url_length = len(url)
    domain_length = len(domain)
    is_domain_ip = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0
    tld = domain.split(".")[-1] if "." in domain and not is_domain_ip else ""
    tld_code = tld_map.get(tld, -1)

    no_subdomain = max(0, len(domain.split(".")) - 2) if ("." in domain and not is_domain_ip) else 0

    has_obfuscation = 1 if "%" in url or "@" in url else 0
    no_obfuscated_char = url.count("%") + url.count("@")
    obfuscation_ratio = no_obfuscated_char / url_length if url_length > 0 else 0

    no_letters = sum(c.isalpha() for c in url)
    letter_ratio = no_letters / url_length if url_length > 0 else 0
    no_digits = sum(c.isdigit() for c in url)
    digit_ratio = no_digits / url_length if url_length > 0 else 0

    no_equals = url.count("=")
    no_qmark = url.count("?")
    no_ampersand = url.count("&")

    special_chars = set("!@#$%^&*()_+-=[]{}|;:'\",.<>?`~")
    no_other_special = sum(1 for c in post_scheme if c in special_chars and c not in ["=", "?", "&"])
    special_char_ratio = sum(1 for c in post_scheme if c in special_chars) / len(post_scheme) if post_scheme else 0

    is_https = 1 if parsed.scheme == "https" else 0

    raw_features = {
        "URLLength": url_length, "DomainLength": domain_length, "IsDomainIP": is_domain_ip,
        "TLD": tld_code, "TLDLength": len(tld), "NoOfSubDomain": no_subdomain,
        "HasObfuscation": has_obfuscation, "NoOfObfuscatedChar": no_obfuscated_char,
        "ObfuscationRatio": obfuscation_ratio, "NoOfLettersInURL": no_letters,
        "LetterRatioInURL": letter_ratio, "NoOfDegitsInURL": no_digits,
        "DegitRatioInURL": digit_ratio, "NoOfEqualsInURL": no_equals,
        "NoOfQMarkInURL": no_qmark, "NoOfAmpersandInURL": no_ampersand,
        "NoOfOtherSpecialCharsInURL": no_other_special, "SpacialCharRatioInURL": special_char_ratio,
        "IsHTTPS": is_https,
    }
    feat_df = pd.DataFrame([raw_features])
    return feat_df[feature_order]


def predict_url(url):
    if not url.strip():
        return "Please enter a URL."
    try:
        feat_df = extract_url_features(url, model.feature_names_in_)
        pred = int(model.predict(feat_df)[0])
        probs = model.predict_proba(feat_df)[0]
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
    description="Paste a URL below to check if it's predicted to be legitimate or phishing. Built with XGBoost on the PhiUSIIL dataset, augmented with Tranco top-sites data for real-world generalization.",
    examples=[
        "https://google.com",
        "https://www.wikipedia.org",
        "http://login.microsoftonline.com.auth-update-portal.info/oauth2",
        "http://secure-paypal-account-verify.tk/login",
    ]
)

import os

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
