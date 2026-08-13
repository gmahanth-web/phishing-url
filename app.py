import os
import re
import joblib
import pandas as pd
import gradio as gr
from urllib.parse import urlparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Load model and TLD mapping
# ---------------------------------------------------------------------------
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


def predict_url(url, history):
    if not url or not url.strip():
        return "⚠️ Please enter a URL to check.", history, history_to_display(history)

    url = url.strip()
    try:
        feat_df = extract_url_features(url, model.feature_names_in_)
        pred = int(model.predict(feat_df)[0])
        probs = model.predict_proba(feat_df)[0]
        confidence = probs[1] if pred == 1 else probs[0]

        if pred == 1:
            result_html = f"""
            <div style="background: linear-gradient(135deg, #0f5132, #146c43); border-radius: 16px; padding: 28px; text-align: center; box-shadow: 0 4px 20px rgba(20,108,67,0.3);">
                <div style="font-size: 48px;">✅</div>
                <div style="font-size: 26px; font-weight: 700; color: #d1f5e0; margin-top: 8px;">Legitimate</div>
                <div style="font-size: 15px; color: #b8e6cb; margin-top: 6px;">Confidence: {confidence*100:.2f}%</div>
                <div style="font-size: 13px; color: #9dd6b5; margin-top: 12px; word-break: break-all;">{url}</div>
            </div>
            """
        else:
            result_html = f"""
            <div style="background: linear-gradient(135deg, #5c1a1a, #842029); border-radius: 16px; padding: 28px; text-align: center; box-shadow: 0 4px 20px rgba(132,32,41,0.35);">
                <div style="font-size: 48px;">🚨</div>
                <div style="font-size: 26px; font-weight: 700; color: #ffd6d6; margin-top: 8px;">Phishing Detected</div>
                <div style="font-size: 15px; color: #ffb3b3; margin-top: 6px;">Confidence: {confidence*100:.2f}%</div>
                <div style="font-size: 13px; color: #ff9999; margin-top: 12px; word-break: break-all;">{url}</div>
            </div>
            """

        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "url": url,
            "result": label_map[pred],
            "confidence": f"{confidence*100:.1f}%"
        }
        history = [entry] + history
        history = history[:10]  # keep last 10

        return result_html, history, history_to_display(history)

    except Exception as e:
        return f"<div style='color:#ff9999;'>Error: {str(e)}</div>", history, history_to_display(history)


def history_to_display(history):
    if not history:
        return "No checks yet this session."
    rows = "".join([
        f"<tr><td style='padding:6px 10px;'>{h['time']}</td>"
        f"<td style='padding:6px 10px; word-break:break-all;'>{h['url']}</td>"
        f"<td style='padding:6px 10px;'>{'✅' if h['result']=='Legitimate' else '🚨'} {h['result']}</td>"
        f"<td style='padding:6px 10px;'>{h['confidence']}</td></tr>"
        for h in history
    ])
    return f"""
    <table style="width:100%; border-collapse:collapse; font-size:13px;">
        <thead>
            <tr style="border-bottom:1px solid #444;">
                <th style="text-align:left; padding:6px 10px;">Time</th>
                <th style="text-align:left; padding:6px 10px;">URL</th>
                <th style="text-align:left; padding:6px 10px;">Result</th>
                <th style="text-align:left; padding:6px 10px;">Confidence</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


custom_css = """
#title { text-align: center; font-size: 32px !important; font-weight: 800 !important; margin-bottom: 4px !important; }
#subtitle { text-align: center; color: #9ca3af !important; margin-bottom: 24px !important; }
.gradio-container { max-width: 720px !important; margin: auto !important; }
footer { visibility: hidden }
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

with gr.Blocks(theme=theme, css=custom_css, title="Phishing URL Detector") as demo:
    gr.HTML("<div id='title'>🛡️ Phishing URL Detector</div>")
    gr.HTML("<div id='subtitle'>Paste any URL to check if it's safe — powered by XGBoost, trained on 230K+ URLs</div>")

    history_state = gr.State([])

    with gr.Row():
        url_input = gr.Textbox(
            placeholder="e.g. https://google.com",
            label="",
            scale=4,
            container=False,
        )
        check_btn = gr.Button("Check URL", variant="primary", scale=1)

    result_output = gr.HTML()

    gr.Examples(
        examples=[
            "https://google.com",
            "https://www.wikipedia.org",
            "http://login.microsoftonline.com.auth-update-portal.info/oauth2",
            "http://secure-paypal-account-verify.tk/login",
        ],
        inputs=url_input,
        label="Try an example",
    )

    with gr.Accordion("📜 Session History", open=False):
        history_display = gr.HTML("No checks yet this session.")

    check_btn.click(
        fn=predict_url,
        inputs=[url_input, history_state],
        outputs=[result_output, history_state, history_display],
    )
    url_input.submit(
        fn=predict_url,
        inputs=[url_input, history_state],
        outputs=[result_output, history_state, history_display],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
