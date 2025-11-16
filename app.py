# Complete app.py — Streamlit + Firebase (email/password auth) + Firestore per-user logs

import streamlit as st
import os
import json
import uuid
import re
from json import JSONDecodeError

# ML libs (keep as in your project; if not needed remove)
import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import cv2

# HTTP for Firebase Auth REST calls
import requests

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore, auth as admin_auth
from google.cloud.firestore import Client as FirestoreClient

# ----------------------------
# Config / Model constants
# ----------------------------
MODEL_FILE = 'indian_food_model.h5'
IMG_SIZE = (224, 224)

# ----------------------------
# Helpers: model loader & image
# ----------------------------
@st.cache_resource
def load_my_model(model_path):
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    else:
        st.error(f"Model '{model_path}' not found.")
        st.stop()

def process_image(image_file):
    img = Image.open(image_file)
    arr = np.array(img)
    try:
        img_rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    except Exception:
        img_rgb = arr
    resized = cv2.resize(img_rgb, IMG_SIZE)
    batch = np.expand_dims(resized, axis=0)
    return preprocess_input(batch)

# ----------------------------
# Firebase initialization (robust)
# ----------------------------
@st.cache_resource
def init_firebase():
    """
    Initialize firebase-admin using either:
      - st.secrets['FIREBASE_JSON'] (preferred on Streamlit Cloud), or
      - local file .streamlit/firebase_key.json (dev).
    This function will attempt to repair common newline issues in the pasted secret.
    """
    try:
        # Try secrets
        fb_secret = None
        try:
            fb_secret = st.secrets.get("FIREBASE_JSON", None)
        except Exception:
            fb_secret = None

        if fb_secret:
            # Accept dict or JSON string
            if isinstance(fb_secret, dict):
                fb_info = fb_secret
            else:
                s = fb_secret
                # Try plain loads
                try:
                    fb_info = json.loads(s)
                except JSONDecodeError:
                    try:
                        fb_info = json.loads(s, strict=False)
                    except Exception:
                        # Attempt to escape actual newlines inside private key
                        try:
                            def _escape_private_key_inner(match):
                                begin = match.group(1)
                                inner = match.group(2)
                                end = match.group(3)
                                inner_escaped = inner.replace("\r\n", "\n").replace("\n", "\\n")
                                return begin + inner_escaped + end
                            pattern = r'(-----BEGIN PRIVATE KEY-----\n)(.*?)(\n-----END PRIVATE KEY-----)'
                            s_fixed = re.sub(pattern, _escape_private_key_inner, s, flags=re.S)
                            fb_info = json.loads(s_fixed)
                        except Exception as e_repair:
                            st.error("Failed to parse FIREBASE_JSON from secrets (repair failed).")
                            st.error(str(e_repair))
                            st.stop()
            cred = credentials.Certificate(fb_info)
        else:
            # Local fallback
            local_path = ".streamlit/firebase_key.json"
            if not os.path.exists(local_path):
                st.error("Firebase key not found. Put service account at `.streamlit/firebase_key.json` or set FIREBASE_JSON in secrets.")
                st.stop()
            cred = credentials.Certificate(local_path)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        return db

    except Exception as e:
        st.error(f"Firebase init error: {e}")
        st.stop()

# ----------------------------
# Firebase Auth (email/password) via REST
# ----------------------------
FIREBASE_API_KEY = st.secrets.get("FIREBASE_API_KEY", None)

_SIGNUP_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
_SIGNIN_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
_TOKEN_REFRESH_URL = "https://securetoken.googleapis.com/v1/token"

def _post_auth(url, payload):
    try:
        resp = requests.post(url, params={"key": FIREBASE_API_KEY}, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError:
        try:
            return {"error": resp.json()}
        except Exception:
            return {"error": "HTTP error during auth request"}
    except Exception as e:
        return {"error": str(e)}

def firebase_signup(email: str, password: str):
    payload = {"email": email, "password": password, "returnSecureToken": True}
    return _post_auth(_SIGNUP_URL, payload)

def firebase_signin(email: str, password: str):
    payload = {"email": email, "password": password, "returnSecureToken": True}
    return _post_auth(_SIGNIN_URL, payload)

def refresh_id_token(refresh_token: str):
    try:
        data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
        resp = requests.post(_TOKEN_REFRESH_URL, params={"key": FIREBASE_API_KEY}, data=data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError:
        try:
            return {"error": resp.json()}
        except Exception:
            return {"error": "HTTP error during token refresh"}
    except Exception as e:
        return {"error": str(e)}

def ensure_id_token():
    """If idToken expired, refresh it using stored refreshToken."""
    if "idToken" not in st.session_state and "refreshToken" in st.session_state:
        ref = st.session_state.get("refreshToken")
        refreshed = refresh_id_token(ref)
        if "error" not in refreshed:
            st.session_state.idToken = refreshed.get("id_token")
            st.session_state.refreshToken = refreshed.get("refresh_token")
            try:
                tokinfo = admin_auth.verify_id_token(st.session_state.idToken)
                st.session_state.user_uid = tokinfo.get("uid")
            except Exception:
                pass

# ----------------------------
# Sidebar Auth UI
# ----------------------------
st.set_page_config(page_title="Cheat Day Tracker", page_icon="🍔", layout="wide")
st.title("🍔 Cheat Day Calorie Tracker 🍕")

st.sidebar.header("Account")
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

auth_mode = st.sidebar.radio("Mode", ("login", "signup", "guest"))

if auth_mode in ("login", "signup"):
    email_in = st.sidebar.text_input("Email", value="", key="auth_email")
    pass_in = st.sidebar.text_input("Password", value="", type="password", key="auth_pass")

if auth_mode == "signup" and st.sidebar.button("Create account"):
    if not FIREBASE_API_KEY:
        st.sidebar.error("FIREBASE_API_KEY missing in Streamlit secrets.")
    elif not email_in or not pass_in:
        st.sidebar.error("Enter email & password.")
    else:
        res = firebase_signup(email_in, pass_in)
        if "error" in res:
            st.sidebar.error(f"Sign-up error: {res['error']}")
        else:
            st.session_state.idToken = res.get("idToken")
            st.session_state.refreshToken = res.get("refreshToken")
            try:
                token_info = admin_auth.verify_id_token(st.session_state.idToken)
                st.session_state.user_uid = token_info.get("uid")
                st.sidebar.success(f"Signed up & logged in as {email_in}")
            except Exception as e:
                st.sidebar.error(f"Auth verify failed: {e}")

if auth_mode == "login" and st.sidebar.button("Sign in"):
    if not FIREBASE_API_KEY:
        st.sidebar.error("FIREBASE_API_KEY missing in Streamlit secrets.")
    elif not email_in or not pass_in:
        st.sidebar.error("Enter email & password.")
    else:
        res = firebase_signin(email_in, pass_in)
        if "error" in res:
            st.sidebar.error(f"Sign-in error: {res['error']}")
        else:
            st.session_state.idToken = res.get("idToken")
            st.session_state.refreshToken = res.get("refreshToken")
            try:
                token_info = admin_auth.verify_id_token(st.session_state.idToken)
                st.session_state.user_uid = token_info.get("uid")
                st.sidebar.success(f"Logged in as {email_in}")
            except Exception as e:
                st.sidebar.error(f"Auth verify failed: {e}")

if auth_mode == "guest" and st.sidebar.button("Continue as guest"):
    st.session_state.user_uid = f"guest_{uuid.uuid4().hex[:12]}"
    st.sidebar.success("Continuing as guest (device only)")

# Show login status + sign out
if st.session_state.get("user_uid"):
    st.sidebar.caption(f"Active uid: {st.session_state.get('user_uid')}")
    if st.sidebar.button("Sign out"):
        for k in ("idToken", "refreshToken", "user_uid"):
            if k in st.session_state:
                del st.session_state[k]
        st.sidebar.success("Signed out")

# ----------------------------
# Initialize Firebase DB (now that auth UI exists)
# ----------------------------
db = init_firebase()

# Ensure token is fresh (optional)
ensure_id_token()

# Decide APP_ID and USER_ID
APP_ID = os.environ.get('__app_id', 'cheat_day_tracker')
USER_ID = st.session_state.get("user_uid", "anon_public")

# ----------------------------
# Firestore helpers (per-user)
# ----------------------------
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

# ----------------------------
# Sidebar: Calorie goal & show user's log
# ----------------------------
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

# ----------------------------
# Main: Upload, classify, log
# (This assumes you have CALORIE_MAP, FRIENDLY_INFO, CLASSES variables imported)
# ----------------------------
try:
    from calorie_database import CALORIE_MAP, FRIENDLY_INFO, CLASSES
except ImportError:
    st.error("Missing calorie_database.py; please add it.")
    st.stop()

st.header("1. Log Your Food")
uploaded = st.file_uploader("Upload your cheat meal:", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="Analyzing...", use_container_width=True)
    with st.spinner("Classifying..."):
        processed = process_image(uploaded)
        model = load_my_model(MODEL_FILE)
        prediction = model.predict(processed)
        idx = int(np.argmax(prediction))
        if idx < len(CLASSES):
            class_name = CLASSES[idx]
            confidence = float(np.max(prediction) * 100)
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
            else:
                st.error(f"Predicted class '{class_name}' not in calorie DB.")
        else:
            st.error("Invalid model prediction index.")

# ----------------------------
# Progress tracker
# ----------------------------
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
