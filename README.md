### 🍔 Cheat Day Calorie Tracker 

A Streamlit-based calorie tracking app that I built using Machine Learning, Firebase, and Python.
It identifies Indian food items from images, estimates calories, lets users log their meals, and tracks their daily intake — all with secure per-user accounts.

This app is live on Streamlit Cloud and fully functional.
Right now, it supports a limited set of Indian food items, but I’ll be expanding the model and calorie database in future updates.

[![Live Demo](https://img.shields.io/badge/Live%20App-Streamlit-brightgreen?style=for-the-badge&logo=streamlit)](https://vishwash-ml-task-05.streamlit.app/)

### 🚀 What This App Does
🔍 Food Recognition (MobileNetV2 Model)

- I trained a custom MobileNetV2-based classifier

- The user uploads an image → the model predicts the dish

- The app maps it to a calorie value using my curated Indian food database

### 📊 Calorie Tracking

- Users can log meals with custom quantity

- Daily calorie limit tracking

- Dynamic progress bar + “over limit” warnings

- Automatic timestamping for every entry

### 👤 User Authentication

- Email + Password sign-up and login (Firebase Auth)

- Each user has isolated data — no overlap

- Guest mode is available for quick testing

### ☁️ Cloud Data Storage

- I use Firestore for storing per-user logs

- Fast reads/writes

- Clean data structure: ```users/{uid}/food_log```

### 🎨 Simple and Clean UI

- Built with Streamlit

- Smooth flow, minimal clicks, responsive layout

## 🧠 Tech Stack

| **Layer** | **Technologies Used** |
|-------|------------------|
| Interface | Streamlit |
| Backend | Python |
| Machine Learning | TensorFlow, Keras, MobileNetV2 |
| Image Processing | OpenCV, Pillow, NumPy |
| Authentication | Firebase Authentication |
| Database | Firestore |
| Deployment | Streamlit Cloud |

### 📂 Project Structure
```
📁 PRODIGY_ML_05
│── app.py                     # Main Streamlit app
│── calorie_database.py        # Calorie + friendly name mappings
│── indian_food_model.h5       # Trained ML model
│── requirements.txt           # Python dependencies
│── packages.txt               # System-level packages
│── task_05.py                 # Dataset splitting utility
│── split_data.py              # Additional dataset tools
│── .gitignore
│── .streamlit/                # Contains secrets.toml (not in repo)
│── data/                      # Dataset (ignored)
```
### 🧠 Model Details

I trained a MobileNetV2-based classifier to recognize Indian foods.
The pipeline:

- Resize image → 224×224

- Preprocess with MobileNetV2 preprocessing

- Predict class → map to Indian dish

- Fetch calorie & unit info from calorie_database.py

### ⚠️ Current Limitation

The model **currently supports a limited set of food items.**
I plan to expand this list and retrain the model with a larger dataset in future versions.

### 🔐 Firebase Setup Used in This Project

1. Enabled Firebase Authentication → Email/Password

2. Set up Firestore in production mode

3. Created a Web App to obtain apiKey

4. Added service account JSON to Streamlit Secrets:
```
FIREBASE_API_KEY = "AIza..."
FIREBASE_JSON = """{ ... }"""
```

The app automatically chooses between local secrets and deployed secrets.

### ▶️ How to Run the App Locally
1. Clone the repo
```
git clone <repo-url>
cd <folder>
```
2. Create a virtual environment
```
python -m venv venv
source venv/bin/activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Add secrets

Create:
```
.streamlit/secrets.toml
```
Paste your Firebase API key and service account JSON.

5. Run the app
```
streamlit run app.py
```
### ☁️ Deployment (How I Deployed It)

- Hosted on Streamlit Cloud

- Secrets added under App Settings → Secrets

- Automatically picks up requirements.txt and packages.txt

- No custom server needed

### 🔥 Why This Project Matters (My Perspective)

This project isn’t just a simple classifier — it’s a full end-to-end product involving:

- Building and training an ML model

- Designing a calorie database

- Integrating Firebase Auth + Firestore

- Writing production-ready Streamlit code

- Managing secrets securely

- Deploying on the cloud

It showcases ML engineering, backend integration, and UI development all in one place.

### 🛠️ What I Want to Add Next

- More food classes (bigger dataset + retraining)

- Google OAuth sign-in

- Weekly analytics dashboard

- Downloadable reports

- Macro breakdown (protein/fats/carbs)

- User profile + preferences

### 🙌 Author

Built by **Vishwash**

A complete Indian food calorie tracker powered by ML + Firebase.


