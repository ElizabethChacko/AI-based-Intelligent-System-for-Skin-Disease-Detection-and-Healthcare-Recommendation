# app.py  ← FINAL VERSION WITH ABCDE RULE + PERFECT DESIGN
import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Skin Lesion Analyzer", page_icon="magnifyingglass", layout="centered")

# ==================== CUSTOM CSS – BEAUTIFUL & PROFESSIONAL ====================
st.markdown("""
<style>
    .big-title {font-size: 56px !important; font-weight: 900; color: #0d47a1; text-align: center; margin-bottom: 0px;}
    .subtitle {font-size: 23px; text-align: center; color: #424242; margin-bottom: 35px;}
    .result-box {padding: 28px; border-radius: 16px; margin: 25px 0; box-shadow: 0 6px 20px rgba(0,0,0,0.12);}
    .low    {background-color: #e8f5e8; border-left: 8px solid #2e7d32;}
    .medium {background-color: #fff8e1; border-left: 8px solid #ff8f00;}
    .high   {background-color: #ffebee; border-left: 8px solid #c62828;}
    .abcde-box {background-color: #e3f2fd; padding: 20px; border-radius: 12px; border-left: 6px solid #1976d2; margin: 20px 0;}
    .disclaimer-box {background-color: #fff3e0; padding: 22px; border-radius: 12px; border: 2px solid #ff9800; margin-top: 30px;}
</style>
""", unsafe_allow_html=True)

# ==================== TITLE ====================
st.markdown('<p class="big-title">Skin Lesion Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered screening • Learn the ABCDE rule • Consult a doctor</p>', unsafe_allow_html=True)

# ==================== DISEASE DATABASE ====================
# ==================== DYNAMIC RESULT SYSTEM (NO HARDCODED RISK) ====================
def get_dynamic_result(pred_class, confidence):
    base_names = {
        'mel': "Melanoma",
        'bcc': "Basal Cell Carcinoma", 
        'akiec': "Actinic Keratosis (Pre-cancer)",
        'bkl': "Benign Keratosis",
        'df': "Dermatofibroma",
        'vasc': "Vascular Lesion",
        'nv': "Common Mole (Melanocytic Nevus)"
    }

    name = base_names[pred_class]

    # DYNAMIC RISK & ADVICE BASED ON PREDICTION + CONFIDENCE
    if confidence < 0.65:
        return {
            "name": name,
            "risk_level": "UNCERTAIN RESULT",
            "color": "medium",
            "info": "AI confidence is too low to give a reliable answer.",
            "advice": [
                "Result is NOT reliable",
                "Please consult a dermatologist for accurate diagnosis",
                "Do not make any decision based on this result"
            ]
        }

    if pred_class == 'mel':
        return {
            "name": name,
            "risk_level": "VERY HIGH RISK",
            "color": "high",
            "info": "Most serious form of skin cancer. Can spread quickly.",
            "advice": [
                "See a dermatologist within 1–2 weeks (sooner if changing)",
                "Do NOT delay — early detection saves lives",
                "Bring this photo and result to your doctor"
            ]
        }

    elif pred_class == 'bcc':
        return {
            "name": name,
            "risk_level": "MODERATE RISK",
            "color": "medium",
            "info": "Most common skin cancer. Grows slowly, rarely spreads.",
            "advice": [
                "Schedule dermatologist visit within 1 month",
                "Usually treated with simple surgery",
                "Excellent cure rate when caught early"
            ]
        }

    elif pred_class == 'akiec':
        return {
            "name": name,
            "risk_level": "MODERATE RISK (Pre-cancer)",
            "color": "medium",
            "info": "Rough patch from sun damage. Can turn into cancer if untreated.",
            "advice": [
                "See dermatologist within 2–3 months",
                "Use SPF 50+ sunscreen daily",
                "Avoid direct sun exposure"
            ]
        }

    else:  # bkl, df, vasc, nv
        return {
            "name": name,
            "risk_level": "NO RISK",
            "color": "low",
            "info": "Completely harmless skin condition.",
            "advice": [
                "No medical treatment needed",
                "Can be removed only for cosmetic reasons",
                "Safe to ignore"
            ]
        }
# ==================== MODEL ====================
@st.cache_resource
def load_model():
    model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=7)
    model.load_state_dict(torch.load("best_model_balanced.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(img):
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.softmax(model(img_t), dim=1)[0]
        conf, idx = torch.max(prob, 0)
    classes = ['akiec','bcc','bkl','df','mel','nv','vasc']
    return classes[idx.item()], conf.item()

# ==================== ABCDE RULE EXPLANATION ====================
with st.expander("What is the ABCDE Rule? (Click to learn how to check moles)", expanded=False):
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 25px;
        border-radius: 16px;
        border-left: 8px solid #1976d2;
        font-family: 'Segoe UI', sans-serif;
    ">
        <h3 style="color: #0d47a1; margin-top: 0; font-size: 22px;">
            ABCDE Rule – How to Spot Dangerous Moles
        </h3>
        <ul style="font-size: 17px; line-height: 2; color: #1e3a5f;">
            <li><strong style="color: #1565c0;">A = Asymmetry</strong> → One half doesn't match the other</li>
            <li><strong style="color: #1565c0;">B = Border</strong> → Irregular, notched, or scalloped edges</li>
            <li><strong style="color: #1565c0;">C = Color</strong> → Multiple colors (brown, black, red, white, blue)</li>
            <li><strong style="color: #1565c0;">D = Diameter</strong> → Larger than 6mm (pencil eraser size)</li>
            <li><strong style="color: #1565c0;">E = Evolving</strong> → Changing in size, shape, or color</li>
        </ul>
        <p style="color: #c62828; font-weight: bold; font-size: 18px; margin: 15px 0 0 0;">
            If your mole shows <u>any</u> of these signs → see a dermatologist immediately!
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================
uploaded = st.file_uploader("Upload a clear, well-lit close-up photo", type=["jpg","jpeg","png","webp","bmp"])

# ==================== MAIN APP (RESULT DISPLAY) ====================
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, width=500, caption="Uploaded Image")

    if st.button("Analyze Lesion", type="primary", use_container_width=True):
        with st.spinner("AI analyzing..."):
            pred_class, confidence = predict(img)
            result = get_dynamic_result(pred_class, confidence)  # ← DYNAMIC RESULT

            st.markdown(f"<div class='result-box {result['color']}'>", unsafe_allow_html=True)
            st.markdown(f"### {result['name']}")
            st.markdown(f"**Risk Level:** {result['risk_level']}  |  **AI Confidence:** {confidence:.1%}")
            st.write(result['info'])
            st.markdown("#### Recommended Next Steps:")
            for tip in result['advice']:
                st.markdown(f"• {tip}")
            st.markdown("</div>", unsafe_allow_html=True)

            # Low confidence warning (extra safety)
            if confidence < 0.7:
                st.warning("Low AI confidence — please consult a dermatologist")
# ==================== FINAL DISCLAIMER ====================
st.markdown("""
<div style="background:#fff8e1; padding:30px; border-radius:20px; border-left:8px solid #ff6f00; font-family:Segoe UI;">
    <p style="color:#e65100; font-size:22px; font-weight:bold; margin:0 0 15px 0;">
        Important Medical Disclaimer
    </p>
    <p style="color:#000000; font-size:17px; line-height:1.9; margin:0;">
        This app is for <strong>educational and preliminary screening only</strong>.<br>
        <span style="color:#c62828; font-weight:bold;">It is NOT a substitute for professional medical diagnosis.</span><br><br>
        Always consult a qualified dermatologist for any skin concern.
    </p>
</div>
""", unsafe_allow_html=True)

st.caption("Made with care for skin health awareness • Version 2025")