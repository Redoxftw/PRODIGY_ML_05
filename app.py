# --- Task 5: 🍔 Cheat Day Calorie Tracker 🍕 ---
#
# --- THIS IS THE FINAL, ROBUST VERSION ---
#
# This script is smart. It knows how to run in two places:
#
# 1. ON STREAMLIT CLOUD (Deployed):
#    It will try to load the key from `st.secrets["firebase_credentials"]`.
#
# 2. ON YOUR PC (Local):
#    If `st.secrets` fails, it will fall back and try to load your
#    local `.streamlit/firebase_key.json` file.
#
# This gives us the best of both worlds.

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

# --- 2. Import Firebase ---
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

# --- 4. NEW: Firebase Initialization ("Smart" Version) ---
@st.cache_resource
def init_firebase():
    """
    Initialize Firebase app using Streamlit Secrets (for deployment)
    or a local key file (for local testing).
    Returns the Firestore database client.
    """
    try:
        # --- METHOD 1: Try to load from Streamlit Secrets (for deployment) ---
        creds_str = st.secrets["firebase_credentials"]
        creds_dict = json.loads(creds_str)
        creds = credentials.Certificate(creds_dict)
        print("Firebase initialized successfully from Streamlit Secrets!")
        
    except KeyError:
        # --- METHOD 2: Fall back to local file (for local testing) ---
        print("Firebase credentials not found in st.secrets. Falling back to local file...")
        KEY_PATH = ".streamlit/firebase_key.json"
        
        if not os.path.exists(KEY_PATH):
            st.error("Firebase Key File Not Found!")
            st.write(f"This app is looking for your secret key at `{KEY_PATH}`.")
            st.write("Please create this file (or add to st.secrets) to run the app.")
            st.stop()
        
        creds = credentials.Certificate(KEY_PATH)
        print("Firebase initialized successfully from local file!")
    
    except Exception as e:
        st.error(f"An error occurred during Firebase initialization: {e}")
        st.stop()

    # Initialize the app if it's not already
    if not firebase_admin._apps:
        firebase_admin.initialize_app(creds)
    
    # Initialize Firestore
    db = firestore.client()
    return db

# --- 5. Database Helper Functions ---
# (These are all the same, they just use the 'db' object)

APP_ID = os.environ.get('__app_id', 'local_app')
USER_ID = os.environ.get('__app_id', 'local_user') # Using __app_id as a unique user ID

def get_log_ref(db: FirestoreClient, app_id: str, user_id: str):
    return db.collection("artifacts").document(app_id).collection("users").document(user_id).collection("food_log")

def get_food_log(db: FirestoreClient):
    try:
        log_ref = get_log_ref(db, APP_ID, USER_ID)
        log_items = log_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        food_log = [item.to_dict() for item in log_items]
        # We need to iterate again to get the document ID
        for i, doc in enumerate(log_items):
             food_log[i]['id'] = doc.id
        return food_log
    except Exception as e:
        st.error(f"Error fetching log: {e}")
        return []


def add_log_item(db: FirestoreClient, item_data):
    item_data["timestamp"] = firestore.SERVER_TIMESTAMP
    log_ref = get_log_ref(db, APP_ID, USER_ID)
    log_ref.add(item_data)

def delete_log_item(db: FirestoreClient, item_id: str):
    log_ref = get_log_ref(db, APP_ID, USER_ID)
    log_ref.document(item_id).delete()

def update_log_item(db: FirestoreClient, item_id: str, new_quantity: int):
    log_ref = get_log_ref(db, APP_ID, USER_ID)
    log_ref.document(item_id).update({"quantity": new_quantity})

def clear_log(db: FirestoreClient):
    log_ref = get_log_ref(db, APP_ID, USER_ID)
    items = log_ref.stream()
    for item in items:
        item.reference.delete()

# --- 6. Image Processing Function (Same as before) ---
def process_image(image_file):
    img = Image.open(image_file)
    img_array = np.array(img)
    try:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    except cv2.error:
        img_rgb = img_array
    resized_img = cv2.resize(img_rgb, IMG_SIZE)
    img_batch = np.expand_dims(resized_img, axis=0)
    processed_batch = preprocess_input(img_batch)
    return processed_batch

# --- 7. Initialize App ---
st.set_page_config(page_title="Cheat Day Tracker", page_icon="🍔", layout="wide")
st.title("🍔 Cheat Day Calorie Tracker 🍕")

db = init_firebase() # This now runs our new "smart" function

# --- 8. Build the App UI (Same as before) ---
# (No changes needed here, it just uses the 'db' object)

# --- === SIDEBAR === ---
st.sidebar.title("🔥 Your Cheat Day HQ 🔥")
st.sidebar.write("Set your goal, then log your food!")

if 'calorie_limit' not in st.session_state:
    st.session_state.calorie_limit = 3000

def update_calorie_limit():
    st.session_state.calorie_limit = st.session_state.limit_input

st.sidebar.header("🎯 Your Calorie Goal")
st.sidebar.number_input(
    "Set your calorie limit for today:",
    min_value=1000,
    max_value=10000,
    value=st.session_state.calorie_limit,
    step=100,
    key="limit_input",
    on_change=update_calorie_limit
)
limit = st.session_state.calorie_limit

st.sidebar.header("📝 Your Food Log")
food_log = get_food_log(db)

if not food_log:
    st.sidebar.info("Your log is empty. Go eat something!")
else:
    total_calories = 0
    for item in food_log:
        item_name = item.get('name', 'Unknown')
        item_calories = item.get('calories', 0)
        item_unit = item.get('unit', 'per serving')
        item_quantity = item.get('quantity', 1)
        item_id = item.get('id', str(uuid.uuid4()))
        
        subtotal = item_calories * item_quantity
        total_calories += subtotal
        
        st.sidebar.markdown(f"**{item_name}**")
        log_col1, log_col2, log_col3 = st.sidebar.columns([3, 2, 1])
        
        with log_col1:
            log_col1.caption(f"{subtotal} kcal ({item_unit})")
        
        with log_col2:
            new_qty = st.number_input(
                "Qty", 
                min_value=1, 
                value=item_quantity, 
                step=1, 
                key=f"qty_{item_id}"
            )
            if new_qty != item_quantity:
                update_log_item(db, item_id, new_qty)
                st.rerun()
        
        with log_col3:
            st.write("")
            if st.button("🗑️", key=f"del_{item_id}"):
                delete_log_item(db, item_id)
                st.rerun()
    
    st.sidebar.divider()
    st.sidebar.subheader(f"Total Intake: {total_calories} kcal")
    
    if st.sidebar.button("Clear Full Log"):
        clear_log(db)
        st.rerun()

# --- === MAIN PAGE === ---
st.header("1. Log Your Food")
uploaded_file = st.file_uploader("Upload a picture of your cheat meal:",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    st.image(uploaded_file, caption="Analyzing...", use_container_width='auto')
    
    with st.spinner("Classifying... 🧠"):
        processed_img_batch = process_image(uploaded_file)
        model = load_my_model(MODEL_FILE)
        prediction = model.predict(processed_img_batch)
        predicted_index = np.argmax(prediction)
        
        if predicted_index < len(CLASSES):
            predicted_class_name = CLASSES[predicted_index]
            confidence = np.max(prediction) * 100
            
            if predicted_class_name in FRIENDLY_INFO:
                friendly_name, unit = FRIENDLY_INFO[predicted_class_name]
                calories = CALORIE_MAP[predicted_class_name]
                
                st.success(f"I think that's a: **{friendly_name}** (Confidence: {confidence:.2f}%)")
                st.info(f"Estimated Calories: **{calories} kcal** *({unit})*")
                
                add_col1, add_col2 = st.columns([1, 1])
                with add_col1:
                    quantity_to_add = st.number_input("Quantity:", min_value=1, value=1, step=1, key="add_qty")
                with add_col2:
                    st.write("")
                    st.write("")
                    if st.button(f"Log {quantity_to_add} x {friendly_name}"):
                        new_item_data = {
                            "name": friendly_name,
                            "calories": calories,
                            "unit": unit,
                            "quantity": quantity_to_add
                        }
                        add_log_item(db, new_item_data)
                        st.success(f"Logged! Your database is updated.")
                        st.rerun()
            else:
                st.error(f"Error: The model predicted '{predicted_class_name}', which is not in the calorie database.")
        else:
            st.error("Error: The model returned an invalid prediction.")

# --- 2. The Progress Tracker (Reads from DB total) ---
st.divider()
st.header("2. Track Your Progress")

total_calories = 0
for item in food_log:
    total_calories += item.get('calories', 0) * item.get('quantity', 1)

st.write(f"You've logged **{total_calories} kcal** out of your **{limit} kcal** goal.")

if limit > 0:
    percent_complete = min(total_calories / limit, 1.0)
else:
    percent_complete = 0.0

st.progress(percent_complete, text=f"{total_calories} / {limit} kcal")

if total_calories > limit:
    st.error(f"🚨 WARNING! You're {total_calories - limit} kcal OVER your limit! 🚨")
elif total_calories > limit * 0.9 and total_calories > 0:
    st.warning(f" Careful! You're almost at your {limit} kcal limit! Only {limit - total_calories} kcal remaining.")
elif total_calories > 0:
    st.success(f"You're doing great! You have {limit - total_calories} kcal remaining. 🎉")
else:
    st.info("Log your first item to start tracking!")