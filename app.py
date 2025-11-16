# --- Task 5: 🍔 Cheat Day Calorie Tracker (Flexible Firebase init) 🍕

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os
import cv2
import json
import uuid

# --- 1. Import My Database ---
try:
    from calorie_database import CALORIE_MAP, FRIENDLY_INFO, CLASSES
except ImportError:
    st.error("FATAL ERROR: 'calorie_database.py' not found!")
    st.write("Please make sure that file is in the same folder as this app.")
    st.stop()

# --- 2. Firebase libs ---
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import Client as FirestoreClient

# --- 3. Setup & Model Loading ---
MODEL_FILE = 'indian_food_model.h5'
IMG_SIZE = (224, 224)

@st.cache_resource
def load_my_model(model_path):
    """Loads our trained .h5 model."""
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print("Food model loaded successfully!")
        return model
    else:
        st.error(f"Error: Model file '{model_path}' not found!")
        st.write("Please run 'python task_05.py' first to train and save the model.")
        st.stop()

# --- 4 Flexible Firebase Initialization (robust to newline issues) ---
@st.cache_resource
def init_firebase():
    """
    Initialize Firebase using:
      1) st.secrets['FIREBASE_JSON'] (preferred)
      2) fallback to .streamlit/firebase_key.json (local dev)
    This version tries to tolerate private_key values pasted with literal newlines
    by escaping them automatically if needed.
    """
    import re
    from json import JSONDecodeError

    try:
        fb_secret = None
        try:
            fb_secret = st.secrets.get("FIREBASE_JSON", None)
        except Exception:
            fb_secret = None

        if fb_secret:
            # if user provided a dict directly in secrets, accept it
            if isinstance(fb_secret, dict):
                fb_info = fb_secret
            else:
                s = fb_secret

                # 1) Try regular strict json parsing
                try:
                    fb_info = json.loads(s)
                except JSONDecodeError:
                    # 2) Try json.loads with strict=False (allows some control chars)
                    try:
                        fb_info = json.loads(s, strict=False)
                    except Exception:
                        # 3) Attempt to repair the private_key by escaping newline chars
                        #    between the BEGIN/END PRIVATE KEY markers.
                        #    This turns actual newlines into the two-character sequence '\n'
                        #    which is valid inside JSON string values.
                        try:
                            def _escape_private_key_inner(match):
                                begin = match.group(1)
                                inner = match.group(2)
                                end = match.group(3)
                                # replace actual newlines with literal backslash-n
                                inner_escaped = inner.replace("\r\n", "\n").replace("\n", "\\n")
                                return begin + inner_escaped + end

                            # This regex finds the block including BEGIN..END and captures the inner content.
                            pattern = r'(-----BEGIN PRIVATE KEY-----\n)(.*?)(\n-----END PRIVATE KEY-----)'
                            s_fixed = re.sub(pattern, _escape_private_key_inner, s, flags=re.S)

                            # Now try to load the repaired string
                            fb_info = json.loads(s_fixed)
                        except Exception as e_repair:
                            st.error("Failed to parse FIREBASE_JSON from secrets (repair attempt failed).")
                            st.error(f"Repair error: {e_repair}")
                            st.stop()

            # Build credentials object from dict
            cred = credentials.Certificate(fb_info)

        else:
            # Local fallback (dev)
            local_path = ".streamlit/firebase_key.json"
            if not os.path.exists(local_path):
                st.error("Firebase key not found. Place your service account at `.streamlit/firebase_key.json` or set FIREBASE_JSON in Streamlit secrets.")
                st.stop()
            cred = credentials.Certificate(local_path)

        # Initialize Firebase app (only once)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        return db

    except Exception as e:
        st.error(f"Error initializing Firebase: {e}")
        st.stop()

# --- 5. Database Helper Functions ---
APP_ID = os.environ.get('__app_id', 'local_app')
USER_ID = os.environ.get('__app_id', 'local_user')

def get_log_ref(db: FirestoreClient, app_id: str, user_id: str):
    return db.collection("artifacts").document(app_id).collection("users").document(user_id).collection("food_log")

def get_food_log(db: FirestoreClient):
    try:
        log_ref = get_log_ref(db, APP_ID, USER_ID)
        items = log_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        food_log = []
        for doc in items:
            d = doc.to_dict()
            d["id"] = doc.id
            food_log.append(d)
        return food_log
    except Exception as e:
        st.error(f"Error fetching log: {e}")
        return []

def add_log_item(db: FirestoreClient, item_data):
    item_data["timestamp"] = firestore.SERVER_TIMESTAMP
    get_log_ref(db, APP_ID, USER_ID).add(item_data)

def delete_log_item(db: FirestoreClient, item_id: str):
    get_log_ref(db, APP_ID, USER_ID).document(item_id).delete()

def update_log_item(db: FirestoreClient, item_id: str, new_quantity: int):
    get_log_ref(db, APP_ID, USER_ID).document(item_id).update({"quantity": new_quantity})

def clear_log(db: FirestoreClient):
    log_ref = get_log_ref(db, APP_ID, USER_ID)
    for item in log_ref.stream():
        item.reference.delete()

# --- 6. Image Processing Function ---
def process_image(image_file):
    img = Image.open(image_file)
    img_array = np.array(img)
    try:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    except cv2.error:
        img_rgb = img_array
    resized_img = cv2.resize(img_rgb, IMG_SIZE)
    img_batch = np.expand_dims(resized_img, axis=0)
    return preprocess_input(img_batch)

# --- 7. Initialize App (UI + DB) ---
st.set_page_config(page_title="Cheat Day Tracker", page_icon="🍔", layout="wide")
st.title("🍔 Cheat Day Calorie Tracker 🍕")

db = init_firebase()

# --- SIDEBAR ---
st.sidebar.title("🔥 Your Cheat Day HQ 🔥")

if 'calorie_limit' not in st.session_state:
    st.session_state.calorie_limit = 3000

def update_calorie_limit():
    st.session_state.calorie_limit = st.session_state.limit_input

st.sidebar.header("🎯 Calorie Goal")
st.sidebar.number_input("Set your calorie limit:", min_value=1000, max_value=10000,
                        value=st.session_state.calorie_limit, step=100,
                        key="limit_input", on_change=update_calorie_limit)
limit = st.session_state.calorie_limit

st.sidebar.header("📝 Your Food Log")
food_log = get_food_log(db)

if not food_log:
    st.sidebar.info("Your log is empty.")
else:
    total_calories = 0
    for item in food_log:
        name = item.get('name', 'Unknown')
        calories = item.get('calories', 0)
        unit = item.get('unit', 'unit')
        qty = item.get('quantity', 1)
        doc_id = item.get('id', uuid.uuid4())

        subtotal = calories * qty
        total_calories += subtotal

        st.sidebar.markdown(f"**{name}**")
        col1, col2, col3 = st.sidebar.columns([3, 2, 1])

        with col1:
            st.caption(f"{subtotal} kcal")

        with col2:
            new_q = st.number_input("Qty", min_value=1, value=qty, step=1, key=f"qty_{doc_id}")
            if new_q != qty:
                update_log_item(db, doc_id, new_q)
                st.rerun()

        with col3:
            if st.button("🗑️", key=f"del_{doc_id}"):
                delete_log_item(db, doc_id)
                st.rerun()

    st.sidebar.subheader(f"Total: {total_calories} kcal")
    if st.sidebar.button("Clear Log"):
        clear_log(db)
        st.rerun()

# --- MAIN PAGE ---
st.header("1. Log Your Food")
uploaded = st.file_uploader("Upload your cheat meal:", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="Analyzing...", use_container_width=True)

    with st.spinner("Classifying..."):
        processed = process_image(uploaded)
        model = load_my_model(MODEL_FILE)
        prediction = model.predict(processed)
        idx = np.argmax(prediction)

        if idx < len(CLASSES):
            class_name = CLASSES[idx]
            confidence = np.max(prediction) * 100

            if class_name in FRIENDLY_INFO:
                friendly, unit = FRIENDLY_INFO[class_name]
                calories = CALORIE_MAP[class_name]

                st.success(f"Identified: **{friendly}** ({confidence:.2f}%)")
                st.info(f"Estimated Calories: **{calories} kcal** ({unit})")

                col1, col2 = st.columns([1, 1])
                with col1:
                    qty = st.number_input("Quantity:", min_value=1, value=1)

                with col2:
                    if st.button(f"Log {qty} × {friendly}"):
                        add_log_item(db, {
                            "name": friendly,
                            "calories": calories,
                            "unit": unit,
                            "quantity": qty
                        })
                        st.success("Logged successfully!")
                        st.rerun()

# --- PROGRESS TRACKER ---
st.divider()
st.header("2. Track Your Progress")

total = sum(i.get('calories', 0) * i.get('quantity', 1) for i in food_log)
st.write(f"You've logged **{total} kcal** out of **{limit} kcal**.")

percent = min(total / max(limit, 1), 1.0)
st.progress(percent, text=f"{total} / {limit} kcal")

if total > limit:
    st.error(f"🚨 You're {total - limit} kcal OVER your limit!")
elif total > 0.9 * limit:
    st.warning(f"Almost there! Only {limit - total} kcal remaining.")
elif total > 0:
    st.success(f"{limit - total} kcal remaining.")
else:
    st.info("Start logging your first meal!")
