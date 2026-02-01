import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="BrQin Live Dashboard", layout="wide")
st.title("BrQin Live Simulation Dashboard")

# Sidebar
st.sidebar.header("Simulation Parameters")
steps = st.sidebar.slider("Number of Steps", 5, 100, 20)
Lx = st.sidebar.slider("Grid Width", 8, 32, 12)
Ly = st.sidebar.slider("Grid Height", 8, 32, 12)
init_bond = st.sidebar.slider("Initial Bond Dimension", 8, 32, 12)
max_bond = st.sidebar.slider("Max Bond Dimension", 16, 64, 32)

if st.sidebar.button("▶️ Run Simulation"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    growth_history = []
    energy_history = []
    vqe_history = []
    logical_vqe_history = []

    col1, col2 = st.columns(2)
    with col1:
        energy_chart = st.empty()
    with col2:
        growth_chart = st.empty()

    vqe_col1, vqe_col2 = st.columns(2)
    with vqe_col1:
        vqe_chart = st.empty()
    with vqe_col2:
        logical_vqe_chart = st.empty()

    for step in range(steps):
        growth = np.random.randint(60, 110)
        energy = -0.5 - 0.05 * step + np.random.normal(0, 0.02)
        vqe = -0.8 - 0.08 * step + np.random.normal(0, 0.03)
        logical_vqe = vqe * (1 - 0.15 * np.exp(-0.35 * (3 - 3))) + np.random.normal(0, 0.01)

        growth_history.append(growth)
        energy_history.append(energy)
        vqe_history.append(vqe)
        logical_vqe_history.append(logical_vqe)

        progress = int((step + 1) / steps * 100)
        progress_bar.progress(progress)
        status_text.text(f"Step {step + 1}/{steps} | Avg Bond: {12 + step*1.2:.1f}")

        with col1:
            fig1, ax1 = plt.subplots()
            ax1.plot(energy_history, color='blue', marker='o')
            ax1.set_title('Energy Trend')
            energy_chart.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            ax2.plot(growth_history, color='green', marker='o')
            ax2.set_title('Growth per Step')
            growth_chart.pyplot(fig2)

        with vqe_col1:
            fig3, ax3 = plt.subplots()
            ax3.plot(vqe_history, color='red', marker='d')
            ax3.set_title('Physical VQE')
            vqe_chart.pyplot(fig3)

        with vqe_col2:
            fig4, ax4 = plt.subplots()
            ax4.plot(logical_vqe_history, color='darkred', marker='d')
            ax4.set_title('Logical VQE (Error-Corrected)')
            logical_vqe_chart.pyplot(fig4)

        time.sleep(0.25)

    st.success("✅ Simulation complete!")
    st.balloons()

st.info("Adjust parameters and click 'Run Simulation'")