"""
Streamlit app: live/uploaded image -> classify item -> add to running bill.

Run with:
    streamlit run 05_app.py

Before running:
    - Make sure best_model.pth and class_names.txt (from 03_train_classifier.py)
      are in the same folder as this script (or update the paths below).
    - Fill in PRICE_DICT with your actual class names and prices.
      Class names must match class_names.txt EXACTLY.
"""

import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ----------------------- CONFIG (edit this) -----------------------
MODEL_PATH = "models/best_model.pth"
CLASS_NAMES_PATH = "models/class_names.txt"
IMG_SIZE = 224

# Fill in with your actual class names (must match class_names.txt exactly)
# and prices in your currency of choice (₹).
#
# IMPORTANT: These are ROUGH, ESTIMATED Indian retail prices based on typical
# MRPs for each brand/category — NOT your actual store's real prices. They're
# here so the demo doesn't show ₹0 for everything. Replace them with your
# real prices before using this for anything beyond a demo.
PRICE_DICT = {
    "Amul Ice Cream - Vanilla Magic 1 L Tub": 250,
    "Appy Fizz Apple Juice -250 ml-": 30,
    "Badshah Afgani Hing": 60,
    "Baygon Mosquito - Fly Killer Spray 625 ml": 220,
    "Bisleri Club Soda - 750 ml": 30,
    "Bisleri Packaged Drinking Water 250 ml": 10,
    "Brooke Bond Red Label Leaf Tea 500 g": 260,
    "Cadbury Bournvita Chocolate Health Drink-1kg-": 420,
    "Catch Ginger Garlic Paste 200 g": 60,
    "Ching-s secret schezwan fried masala": 55,
    "Chings Secret Veg Hakka Noodles 140 g Pouch": 30,
    "Coca-Cola-600 ml-": 40,
    "Dabur Dant Rakshak Ayurvedic Paste 175 g -Pack of 2-": 180,
    "Dettol Bathing Soap Bar - Original 125g": 45,
    "Everest Chaat Masala 50 g": 45,
    "Everest Chicken Masala 100 g": 70,
    "Everest Pav Bhaji Masala 100 g": 65,
    "Exo Touch - Shine Anti-Bacterial Round Dishwash Bar 700 g": 60,
    "Fortune Sunlite Refined Sunflower Oil 5 L": 750,
    "Gokul Milk 500ml": 30,
    "Gold Oil": 160,
    "Govardhan Pure cow Ghee -1kg-": 620,
    "Govardhan Pure cow Ghee -200ml-": 160,
    "HIT Mosquito - Fly Killer Spray700 ml": 210,
    "Kimia Dates": 180,
    "Knorr Pizza Pasta Sauce": 95,
    "LG Compounded Hing 100 g": 90,
    "LG Compounded Hing Powder 50 g": 55,
    "MAGGI Pichkoo - Rich Tomato Ketchup 80 g Pouch": 15,
    "Maggi 2-Minute Masala Instant Noodles 280 g": 55,
    "Nivea Fresh Natural Deodorant 150 ml": 220,
    "Organic India Tulsi Green Tea 100 g": 210,
    "Parachute Coconut Oil 300 ml": 150,
    "Parachute Coconut Oil 600 ml - Bottle": 280,
    "Pintola High Protein Peanut Butter-510 gm-": 380,
    "Pond-s Dreamflower Pink Lily Fragrant Talc 400 g": 180,
    "Smith - Jones Ginger Garlic Paste 200 g": 55,
    "Tata Salt Vacuum Evaporated Iodised Salt 1 kg Pouch": 28,
    "Tropicana Fruit Juice - Delight Guava1 L": 110,
}
# --------------------------------------------------------------------


@st.cache_resource
def load_class_names(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


@st.cache_resource
def load_model(model_path, num_classes, device):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


def predict(model, transform, img, class_names, device):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())
    return class_names[pred_idx], confidence


def main():
    st.set_page_config(page_title="Self-Checkout Classifier", layout="centered")
    st.title("Self-Checkout Classifier")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(CLASS_NAMES_PATH):
        st.error(
            f"Missing model files. Expected '{MODEL_PATH}' and '{CLASS_NAMES_PATH}' "
            f"in the same folder as this script. Run 03_train_classifier.py first."
        )
        st.stop()

    class_names = load_class_names(CLASS_NAMES_PATH)
    model = load_model(MODEL_PATH, len(class_names), device)
    transform = get_transform(IMG_SIZE)

    missing_prices = [c for c in class_names if c not in PRICE_DICT]
    if missing_prices:
        st.warning(
            f"No price set for {len(missing_prices)} class(es): {missing_prices}. "
            f"They'll be added to the bill at ₹0 until you fill in PRICE_DICT."
        )

    if "bill" not in st.session_state:
        st.session_state.bill = []  # list of (class_name, price, confidence)

    st.subheader("Scan an item")
    tab1, tab2 = st.tabs(["Camera", "Upload"])

    uploaded = None
    with tab1:
        cam_img = st.camera_input("Take a photo of the item")
        if cam_img is not None:
            uploaded = cam_img
    with tab2:
        file_img = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])
        if file_img is not None:
            uploaded = file_img

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        cls_name, confidence = predict(model, transform, img, class_names, device)
        price = PRICE_DICT.get(cls_name, 0)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, width=150)
        with col2:
            st.success(f"Detected: **{cls_name}** ({confidence*100:.1f}% confidence)")
            st.write(f"Price: ₹{price}")

            if st.button("Add to bill"):
                st.session_state.bill.append((cls_name, price, confidence))
                st.rerun()

    st.divider()
    st.subheader("Current bill")

    if not st.session_state.bill:
        st.info("No items added yet.")
    else:
        total = 0
        for i, (item, price, conf) in enumerate(st.session_state.bill):
            c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
            c1.write(item)
            c2.write(f"₹{price}")
            c3.write(f"{conf*100:.0f}% conf.")
            if c4.button("✕", key=f"remove_{i}"):
                st.session_state.bill.pop(i)
                st.rerun()
            total += price

        st.markdown(f"### Total: ₹{total}")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Checkout / Finalize"):
                st.balloons()
                st.success(f"Checkout complete. Final total: ₹{total}")
        with col_b:
            if st.button("Clear bill"):
                st.session_state.bill = []
                st.rerun()


if __name__ == "__main__":
    main()