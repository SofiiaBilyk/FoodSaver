# FoodSaverV0.py

"""
Food Saver – AI for Reducing Food Waste in Student Communities

Core Features:
- Scan or input fridge contents
- Predict spoilage timelines
- Recommend recipes based on soon-to-expire items
- Suggest when to freeze, share, or donate food
- Learn user habits to optimize purchase and meal suggestions
"""

import streamlit as st
import datetime
from PIL import Image
import requests
import io
import base64
import tensorflow as tf
import numpy as np
import google.generativeai as genai
from streamlit_carousel import carousel
import json
from transformers import pipeline

# --- Mock Data and Utilities ---

# Example food database with shelf life (in days)
FOOD_DB = {
    "milk": 7,
    "eggs": 21,
    "chicken": 3,
    "lettuce": 5,
    "cheese": 14,
    "tomato": 5,
    "bread": 4,
    "apple": 14,
    "yogurt": 10,
}

# Example recipes
RECIPES = [
    {"name": "Omelette", "ingredients": ["eggs", "cheese", "tomato"]},
    {"name": "Chicken Salad", "ingredients": ["chicken", "lettuce", "tomato"]},
    {"name": "Grilled Cheese", "ingredients": ["bread", "cheese"]},
    {"name": "Fruit Salad", "ingredients": ["apple", "yogurt"]},
]

# --- Core Classes ---

class FoodItem:
    def __init__(self, name, added_date=None, expiry_date=None):
        self.name = name
        self.added_date = added_date or datetime.date.today()
        self.expiry_date = expiry_date
        self.shelf_life = FOOD_DB.get(name, 7)  # Default 7 days if unknown

    def days_left(self):
        if self.expiry_date:
            # Calculate days left based on expiry date
            days = (self.expiry_date - datetime.date.today()).days
        else:
            # Calculate based on shelf life
            elapsed = (datetime.date.today() - self.added_date).days
            days = self.shelf_life - elapsed
        return max(days, 0)

    def is_expired(self):
        return self.days_left() <= 0

# --- AI Logic (Mocked/Prototype) ---

def soon_to_expire(days=3):
    fridge = get_fridge()
    return [item for item in fridge if 0 < item.days_left() <= days]

def predict_spoilage():
    fridge = get_fridge()
    st.subheader("Spoilage Predictions")
    for item in fridge:
        st.write(f"- {item.name}: {item.days_left()} days left")

def recommend_recipes():
    fridge = get_fridge()
    available = set(item.name for item in fridge if not item.is_expired())
    suggestions = []
    for recipe in RECIPES:
        if set(recipe["ingredients"]).issubset(available):
            suggestions.append(recipe["name"])
    st.subheader("Recipe Suggestions")
    if suggestions:
        for r in suggestions:
            st.write(f"- {r}")
    else:
        st.write("No full recipes available. Try using soon-to-expire items in simple meals.")

def suggest_actions():
    st.subheader("Action Suggestions")
    for item in soon_to_expire():
        st.write(f"- {item.name} is expiring soon. Suggest: use, freeze, share, or donate.")

def optimize_purchases():
    history = get_history()
    wasted = [item.name for item, action in history if action == "wasted"]
    st.subheader("Purchase Optimization")
    if wasted:
        for food in set(wasted):
            st.write(f"- Consider buying less {food} (often wasted).")
    else:
        st.write("No waste detected yet. Keep it up!")

def upload_and_show_image():
    path = input("Enter the path to your fridge/products image: ").strip()
    try:
        img = Image.open(path)
        img.show()
        print("Image uploaded and displayed successfully!")
        # Here you could add code to process the image with a food recognition model
    except Exception as e:
        print(f"Error opening image: {e}")

def get_fridge():
    if "fridge" not in st.session_state:
        st.session_state.fridge = []
    return st.session_state.fridge

def get_history():
    if "history" not in st.session_state:
        st.session_state.history = []
    return st.session_state.history

def add_food(food_name, expiry_date=None):
    if not food_name:
        st.warning("Please enter a food name.")
        return
    fridge = get_fridge()
    fridge.append(FoodItem(food_name, expiry_date=expiry_date))
    if expiry_date:
        st.success(f"Added {food_name} to fridge (expires on {expiry_date.strftime('%Y-%m-%d')})")
    else:
        st.success(f"Added {food_name} to fridge")

def remove_food(food_name, action="used"):
    fridge = get_fridge()
    history = get_history()
    for item in fridge:
        if item.name == food_name:
            fridge.remove(item)
            history.append((item, action))
            st.success(f"{action.capitalize()} {food_name}.")
            return
    st.warning(f"{food_name} not found in fridge.")

def detect_food_in_image(image, api_key, model="food-detection-3q9ga/1"):
    # Convert PIL image to bytes
    buffered = io.BytesIO()
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Roboflow API endpoint
    url = f"https://detect.roboflow.com/{model}"
    params = {
        "api_key": api_key,
        "confidence": 20,  # minimum confidence threshold
    }
    response = requests.post(
        url,
        params=params,
        data=img_str,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    if response.status_code == 200:
        predictions = response.json().get("predictions", [])
        detected = [pred["class"] for pred in predictions]
        return list(set(detected))  # unique food items
    else:
        st.error("Food detection failed. Check your API key and model name.")
        return []

def get_roboflow_api_key():
    try:
        with open("roboflow_key.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error("Roboflow API key file not found. Please create 'roboflow_key.txt' in your project folder.")
        return None

# Load the model once (outside the function for efficiency)
@st.cache_resource
def load_model():
    return tf.keras.applications.mobilenet_v2.MobileNetV2(weights="imagenet")

model = load_model()
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def detect_top_food_in_image(image):
    image = image.convert("RGB").resize((224, 224))
    x = np.array(image)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=1)[0]
    return decoded[0]

def get_gemini_api_key():
    try:
        with open(".env", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error("Please create gemini API key")
        return None

def get_gemini_recipes(fridge_items):
    try:
        # Read API key from file
        with open('.env', 'r') as file:
            api_key = file.read().strip()
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create the prompt
        items_text = "\n".join([f"- {item.name} ({item.days_left()} days left)" for item in fridge_items])
        prompt = f"""
        Based on these ingredients in my fridge:
        {items_text}

        Generate exactly 4 recipes that:
        1. Use as many available ingredients as possible
        2. Prioritize ingredients that will expire soon

        Format each recipe exactly like this example:
        
        Title: Tomato & Cheese Egg Bites (Mini Frittatas)

        Ingredients from Fridge:
        - Tomato
        - Eggs
        - Cheese

        Additional Ingredients Needed:
        - Salt
        - Pepper
        - Optional: Milk or Cream (a splash)
        - Muffin tin
        - Oil or cooking spray

        Cooking Instructions:
        1. Preheat oven to 350°F (175°C). Grease or spray a muffin tin with oil.
        2. Dice the tomato.
        3. Whisk eggs in a bowl with a splash of milk/cream (optional), salt and pepper.
        4. Stir in the diced tomato and shredded cheese.
        5. Pour the egg mixture into the muffin tin cups, filling each about ¾ full.
        6. Bake for 20-25 minutes, or until the egg bites are set and slightly golden.
        7. Let cool slightly before removing from the muffin tin. Serve warm.

        ---

        Please provide exactly 4 different recipes following this exact format, separated by '---'.
        Make sure each recipe is practical and uses different combinations of the available ingredients.
        """
        
        # Generate the response
        response = model.generate_content(prompt)
        
        # Return the response text
        return response.text
        
    except FileNotFoundError:
        st.error("❌ Error: .env file not found. Please create the file with your API key.")
        return None
    except Exception as e:
        st.error(f"❌ Gemini API connection failed: {e}")
        return None

def show_recipes(recipes_text):
    if recipes_text:
        # Split the text into recipes (assuming recipes are separated by ---)
        recipes = [r.strip() for r in recipes_text.split('---') if r.strip()]
        
        # Create tabs for each recipe
        tabs = st.tabs([f"Recipe {i+1}" for i in range(len(recipes))])
        
        # Display each recipe in its tab
        for tab, recipe in zip(tabs, recipes):
            with tab:
                st.markdown(recipe)
                # Add Instacart button if additional ingredients are needed
                if "Additional Ingredients Needed:" in recipe:
                    st.markdown("---")
                    st.link_button("Get Additional Ingredients 🛒", url="https://www.instacart.com", type="primary")

# --- Streamlit UI ---

st.title("🥕 Food Saver – Reduce Food Waste!")

st.sidebar.header("Fridge Management")
food_to_add = st.sidebar.text_input("Add food item (e.g. milk, eggs):")
expiry_date = st.sidebar.date_input("Expiry date (optional)", 
    value=None,
    min_value=datetime.date.today(),
    help="Leave empty to use default shelf life"
)
if st.sidebar.button("Add Food"):
    if food_to_add:
        # Only use expiry_date if user selected a date
        if expiry_date and expiry_date != datetime.date.today():
            add_food(food_to_add.strip().lower(), expiry_date=expiry_date)
        else:
            add_food(food_to_add.strip().lower())

food_to_remove = st.sidebar.text_input("Remove food item:")
action = st.sidebar.selectbox("Action", ["used", "wasted", "shared", "donated"])
if st.sidebar.button("Remove Food"):
    remove_food(food_to_remove.strip().lower(), action)

st.sidebar.markdown("---")
st.sidebar.header("Upload Fridge Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Fridge Image", use_column_width=True)
    st.success("Image uploaded!")

    st.subheader("AI Food Detection")
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    results = classifier(image)
    top_pred = detect_top_food_in_image(image)
    class_name = top_pred[1]
    score = top_pred[2] * 100

    st.write(f"**Detected food:** {class_name} ({score:.2f}%)")

    if st.button("Add Detected Food to Fridge"):
        add_food(class_name.lower())
        st.success(f"Added {class_name} to fridge!")

st.header("Your Fridge")
fridge = get_fridge()
if fridge:
    for item in fridge:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"- {item.name}: {item.days_left()} days left")
            if item.days_left() <= 3 and item.days_left() > 0:
                st.markdown(f"<span style='color: red'>⚠️ This item could be donated to avoid food waste!</span>", unsafe_allow_html=True)
                if st.button(f"Donate {item.name}", key=f"donate_{item.name}"):
                    remove_food(item.name, action="donated")
                    st.success("Thank you for your donation! 🙏")
                    st.rerun()
else:
    st.write("Your fridge is empty. Add some food!")

if st.button("Suggest Recipes"):
    fridge = get_fridge()
    if fridge:
        with st.spinner("Generating recipe suggestions..."):
            recipes = get_gemini_recipes(fridge)
            if recipes:
                show_recipes(recipes)
            else:
                st.info("No recipes generated. Please check your API key and try again.")
    else:
        st.warning("Your fridge is empty! Add some ingredients first.")

print(tf.__version__)


