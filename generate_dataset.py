"""
generate_dataset.py
--------------------
Creates a synthetic sample dataset that mimics real-world accessibility
audit data. This dataset is used to train the AI (Random Forest) model
that classifies a design as Accessible / Partially Accessible / Needs
Improvement.

Run this file directly to (re)create sample_dataset.csv:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd

# Make results repeatable
np.random.seed(42)

N_SAMPLES = 500

# ----------------------------------------------------------------------
# 1. Randomly generate raw accessibility feature values
# ----------------------------------------------------------------------
font_size = np.random.randint(8, 33, N_SAMPLES)                 # px, 8-32
color_contrast = np.round(np.random.uniform(1.0, 21.0, N_SAMPLES), 2)  # contrast ratio
alt_text = np.random.choice(["Yes", "Partial", "No"], N_SAMPLES, p=[0.45, 0.25, 0.30])
keyboard_nav = np.random.choice(["Yes", "No"], N_SAMPLES, p=[0.55, 0.45])
screen_reader = np.random.choice(["Yes", "Partial", "No"], N_SAMPLES, p=[0.4, 0.3, 0.3])
form_labels = np.random.choice(["Yes", "Partial", "No"], N_SAMPLES, p=[0.5, 0.25, 0.25])


# ----------------------------------------------------------------------
# 2. Scoring function (same logic used inside the Streamlit app)
#    Total = 100 points, split across 6 accessibility factors
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


def score_yes_partial_no(value, full, partial):
    if value == "Yes":
        return full
    elif value == "Partial":
        return partial
    return 0


def score_keyboard(value):
    return 20 if value == "Yes" else 0


rows = []
for i in range(N_SAMPLES):
    s_font = score_font_size(font_size[i])
    s_contrast = score_contrast(color_contrast[i])
    s_alt = score_yes_partial_no(alt_text[i], 15, 7)
    s_keyboard = score_keyboard(keyboard_nav[i])
    s_screen = score_yes_partial_no(screen_reader[i], 15, 7)
    s_form = score_yes_partial_no(form_labels[i], 15, 7)

    total_score = s_font + s_contrast + s_alt + s_keyboard + s_screen + s_form

    # Add a small amount of random noise so the ML model has to learn
    # patterns rather than simply memorising the scoring formula.
    noisy_score = total_score + np.random.normal(0, 4)

    if noisy_score >= 85:
        label = "Accessible"
    elif noisy_score >= 60:
        label = "Partially Accessible"
    else:
        label = "Needs Improvement"

    rows.append([
        font_size[i], color_contrast[i], alt_text[i], keyboard_nav[i],
        screen_reader[i], form_labels[i], total_score, label
    ])

df = pd.DataFrame(rows, columns=[
    "Font_Size", "Color_Contrast", "Alt_Text", "Keyboard_Navigation",
    "Screen_Reader", "Form_Labels", "Accessibility_Score", "Label"
])

if __name__ == "__main__":
    df.to_csv("sample_dataset.csv", index=False)
    print(f"sample_dataset.csv created with {len(df)} rows.")
    print(df["Label"].value_counts())
