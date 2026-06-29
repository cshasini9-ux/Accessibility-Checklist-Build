"""
model_utils.py
---------------
Helper functions for:
  1. Encoding categorical accessibility answers into numbers
  2. Calculating the rule-based accessibility score (0-100)
  3. Training a Random Forest classifier on the sample dataset
  4. Using the trained model to predict an accessibility category
  5. Generating simple text recommendations based on user input

Keeping all the "ML logic" in one file makes app.py easier to read.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------------------------------------------------
# Encoding maps (Yes/Partial/No -> numbers) so the ML model can use them
# ----------------------------------------------------------------------
YES_PARTIAL_NO_MAP = {"Yes": 2, "Partial": 1, "No": 0}
YES_NO_MAP = {"Yes": 1, "No": 0}


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical Yes/Partial/No columns into numeric columns."""
    encoded = df.copy()
    encoded["Alt_Text"] = encoded["Alt_Text"].map(YES_PARTIAL_NO_MAP)
    encoded["Keyboard_Navigation"] = encoded["Keyboard_Navigation"].map(YES_NO_MAP)
    encoded["Screen_Reader"] = encoded["Screen_Reader"].map(YES_PARTIAL_NO_MAP)
    encoded["Form_Labels"] = encoded["Form_Labels"].map(YES_PARTIAL_NO_MAP)
    return encoded


# ----------------------------------------------------------------------
# Rule-based scoring (out of 100) -- used for the score card / charts
# ----------------------------------------------------------------------
def score_font_size(px):
    if px >= 16:
        return 15
    elif px >= 12:
        return 7
    return 0


def score_contrast(ratio):
    if ratio >= 4.5:
        return 20
    elif ratio >= 3.0:
        return 10
    return 0


def score_yes_partial_no(value, full=15, partial=7):
    if value == "Yes":
        return full
    elif value == "Partial":
        return partial
    return 0


def score_keyboard(value):
    return 20 if value == "Yes" else 0


def calculate_accessibility_score(font_size, color_contrast, alt_text,
                                   keyboard_nav, screen_reader, form_labels):
    """
    Returns a dict with the breakdown of points per factor + the total
    accessibility score out of 100.
    """
    breakdown = {
        "Font Size": score_font_size(font_size),
        "Color Contrast": score_contrast(color_contrast),
        "Image Alt Text": score_yes_partial_no(alt_text),
        "Keyboard Navigation": score_keyboard(keyboard_nav),
        "Screen Reader Support": score_yes_partial_no(screen_reader),
        "Form Labels": score_yes_partial_no(form_labels),
    }
    total = sum(breakdown.values())
    return breakdown, total


# ----------------------------------------------------------------------
# Train the Random Forest model (cached by Streamlit in app.py)
# ----------------------------------------------------------------------
def train_model(df: pd.DataFrame):
    """
    Trains a RandomForestClassifier on the sample dataset and
    returns the trained model, the label encoder, and the test accuracy.
    """
    feature_cols = [
        "Font_Size", "Color_Contrast", "Alt_Text",
        "Keyboard_Navigation", "Screen_Reader", "Form_Labels"
    ]

    encoded_df = encode_features(df)

    X = encoded_df[feature_cols]
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, le, accuracy


def predict_category(model, label_encoder, font_size, color_contrast,
                      alt_text, keyboard_nav, screen_reader, form_labels):
    """Use the trained model to classify a single new entry."""
    input_df = pd.DataFrame([{
        "Font_Size": font_size,
        "Color_Contrast": color_contrast,
        "Alt_Text": alt_text,
        "Keyboard_Navigation": keyboard_nav,
        "Screen_Reader": screen_reader,
        "Form_Labels": form_labels,
    }])

    encoded_input = encode_features(input_df)
    prediction = model.predict(encoded_input)
    return label_encoder.inverse_transform(prediction)[0]


# ----------------------------------------------------------------------
# Recommendations generator
# ----------------------------------------------------------------------
def generate_recommendations(font_size, color_contrast, alt_text,
                              keyboard_nav, screen_reader, form_labels):
    """Return a list of plain-English suggestions based on weak spots."""
    tips = []

    if font_size < 16:
        tips.append("Increase the base font size to at least 16px for better readability.")

    if color_contrast < 4.5:
        tips.append("Improve color contrast to meet the WCAG AA minimum ratio of 4.5:1.")

    if alt_text == "No":
        tips.append("Add descriptive alternative text (alt text) to all images.")
    elif alt_text == "Partial":
        tips.append("Review images missing alt text and add descriptions where needed.")

    if keyboard_nav == "No":
        tips.append("Enable full keyboard navigation (Tab, Enter, Arrow keys) for all interactive elements.")

    if screen_reader == "No":
        tips.append("Add ARIA labels and semantic HTML so screen readers can interpret the content correctly.")
    elif screen_reader == "Partial":
        tips.append("Audit screen reader support for dynamic or interactive components.")

    if form_labels == "No":
        tips.append("Add clear <label> tags to every form field.")
    elif form_labels == "Partial":
        tips.append("Ensure all form fields (not just some) have associated labels.")

    if not tips:
        tips.append("Great job! No major accessibility issues detected. Keep monitoring as content changes.")

    return tips
