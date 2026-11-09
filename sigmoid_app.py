# app_sigmoid.py
# Streamlit app: Visualise the Sigmoid activation function

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Activation Function", layout="wide")

st.title("2. Sigmoid Activation Function")
st.write(
    """
    The **Sigmoid** (logistic) activation function is defined as:

    $$\\sigma(x) = \\frac{1}{1 + e^{-x}}$$

    It squashes any real input into the range **(0, 1)**, making it useful for
    output layers of binary classification models. Its derivative is
    $\\sigma'(x) = \\sigma(x)(1 - \\sigma(x))$.
    """
)

# --- Sidebar controls ---
st.sidebar.header("Settings")
x_min = st.sidebar.slider("x-axis minimum", -20.0, 0.0, -6.0, 0.5)
x_max = st.sidebar.slider("x-axis maximum", 0.0, 20.0, 6.0, 0.5)
n_points = st.sidebar.slider("Number of points", 100, 2000, 400, 50)
show_derivative = st.sidebar.checkbox("Show derivative", value=False)

# --- Compute values ---
x = np.linspace(x_min, x_max, n_points)
y = 1.0 / (1.0 + np.exp(-x))

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color="royalblue", linewidth=2.5, label="σ(x) = 1 / (1 + e^-x)")

if show_derivative:
    dy = y * (1 - y)
    ax.plot(x, dy, color="darkorange", linewidth=2, linestyle="--",
            label="σ'(x) = σ(x)(1 - σ(x))")

ax.axhline(0, color="black", linewidth=0.6)
ax.axhline(1, color="grey", linewidth=0.6, linestyle=":")
ax.axhline(0.5, color="grey", linewidth=0.6, linestyle=":")
ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Sigmoid Activation Function")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

st.pyplot(fig)

# --- Property table ---
st.subheader("Properties")
st.table({
    "Property": ["Range", "Derivative", "Zero-centred?", "Vanishing gradient?"],
    "Value": [
        "(0, 1)",
        "σ(x)(1 − σ(x))",
        "No",
        "Yes (for large |x|)",
    ],
})

# --- Interactive single-value calculator ---
st.subheader("Try a value")
xv = st.number_input("Enter x", value=0.0, step=0.1)
val = 1.0 / (1.0 + np.exp(-xv))
st.write(f"**Sigmoid({xv}) = {val:.6f}**")