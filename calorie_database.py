# --- calorie_database.py (Version 2) ---
# I've added a 'CLASSES' list at the bottom.
# This list MUST be in the same order that Keras finds the folders.
# We will verify this when we run the trainer script.

CALORIE_MAP = {
    "biryani": 350,
    "chole_bhature": 450,
    "dal": 180,
    "dhokla": 120,
    "dosa": 300,
    "jalebi": 150,
    "kofta": 320,
    "naan": 260,
    "pakora": 170,
    "paneer": 280,
    "pani_puri": 150,
    "vadapav": 300
}

FRIENDLY_INFO = {
    "biryani": ("Biryani", "per bowl"),
    "chole_bhature": ("Chole Bhature", "per plate"),
    "dal": ("Dal", "per bowl"),
    "dhokla": ("Dhokla", "per 2 pieces"),
    "dosa": ("Dosa", "per piece"),
    "jalebi": ("Jalebi", "per piece"),
    "kofta": ("Kofta", "per bowl"),
    "naan": ("Naan", "per piece"),
    "pakora": ("Pakora", "per 2-3 pieces"),
    "paneer": ("Paneer", "per serving"),
    "pani_puri": ("Pani Puri", "per plate"),
    "vadapav": ("Vadapav", "per piece")
}

# --- THIS IS THE NEW PART ---
# This list is the "source of truth".
# The output of our model will be an index (0-11).
# We will use this list to map that index back to a name.
# e.g., Index 0 = "biryani"
#
# !! CRITICAL !!: This list MUST match the class order
# that your 'task_05.py' script prints out.
# (The script prints: "Found 12 classes: ['biryani', 'chole_bhature', ...]")
# I've used the 12 folders you showed me, in alphabetical order.
CLASSES = [
    'biryani',
    'chole_bhature',
    'dal',
    'dhokla',
    'dosa',
    'jalebi',
    'kofta',
    'naan',
    'pakora',
    'paneer',
    'pani_puri',
    'vadapav'
]