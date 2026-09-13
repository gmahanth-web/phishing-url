import os
import re
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from urllib.parse import urlparse

app = Flask(__name__)

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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = (data or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        feat_df = extract_url_features(url, model.feature_names_in_)
        pred = int(model.predict(feat_df)[0])
        probs = model.predict_proba(feat_df)[0]
        return jsonify({
            "url": url,
            "prediction": label_map[pred],
            "prob_phishing": round(float(probs[0]), 4),
            "prob_legitimate": round(float(probs[1]), 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
