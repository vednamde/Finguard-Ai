import base64
import streamlit as st
from pathlib import Path
import pandas as pd
import joblib
from fpdf import FPDF
import io

# ===============================
# ✅ App Configuration
# ===============================
st.set_page_config(page_title="FinGuard AI", page_icon="💹", layout="centered")

# Load trained model
model = joblib.load("../models/financial_health_xgb.pkl")

# ===============================
# 🖼️ Header with Animated Logo
# ===============================
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
logo_path = current_dir / "logo.png"

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

if logo_path.exists():
    logo_base64 = get_base64_image(logo_path)
    st.markdown(f"""
        <style>
            .header-container {{
                background: linear-gradient(120deg, #00c853, #00b8d4, #1de9b6);
                background-size: 400% 400%;
                animation: gradientShift 8s ease infinite;
                padding: 35px;
                border-radius: 16px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                margin-bottom: 30px;
            }}
            @keyframes gradientShift {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}
            .header-container img {{
                width: 280px;
                border-radius: 12px;
                box-shadow: 0 0 20px rgba(255,255,255,0.5);
            }}
        </style>

        <div class="header-container">
            <img src="data:image/png;base64,{logo_base64}" alt="FinGuard AI Logo">
        </div>
    """, unsafe_allow_html=True)

# ===============================
# 🧾 App Description
# ===============================
st.markdown("""
<div style="background-color:#1e1e1e; padding:15px; border-radius:10px; color:#e0f2f1; text-align:center;">
This app uses <b>Machine Learning</b> to determine whether a company is 
<b style='color:#00e676;'>Financially Healthy</b> or 
<b style='color:#ff5252;'>At Risk of Distress</b> based on financial metrics.
</div>
""", unsafe_allow_html=True)

# ===============================
# 📂 File Upload Section
# ===============================
st.markdown("### 📂 Upload Company Financial Data")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("✅ File successfully uploaded!")
    st.dataframe(df.head(), use_container_width=True)

    # Predictions
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    results = pd.DataFrame({
        "Company": df.get("Company", [f"Company {i+1}" for i in range(len(predictions))]),
        "Predicted Status": ["Healthy" if p == 0 else "Distressed" for p in predictions],
        "Risk Probability (%)": (probabilities * 100).round(2)
    })

    # KPI Summary
    healthy = (results["Predicted Status"] == "Healthy").sum()
    distressed = (results["Predicted Status"] == "Distressed").sum()
    avg_health = 100 - results["Risk Probability (%)"].mean()

    st.markdown("### 📊 Company Financial Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Healthy Companies", healthy)
    col2.metric("At Risk Companies", distressed)
    col3.metric("Avg. Health Score", f"{avg_health:.2f}%")

    # Prediction Results
    st.subheader("🧠 Prediction Results")
    for _, row in results.iterrows():
        color = "green" if row["Predicted Status"] == "Healthy" else "red"
        st.markdown(
            f"<div style='background-color:rgba(255,255,255,0.05); padding:10px; border-radius:10px; margin-bottom:8px;'>"
            f"<span style='color:{color}; font-size:18px;'>"
            f"<strong>{row['Company']}</strong>: {row['Predicted Status']} "
            f"— <em>Risk Probability:</em> {row['Risk Probability (%)']}%"
            f"</span></div>",
            unsafe_allow_html=True
        )

    # Health Score Chart
    results["Financial Health Score (0-100)"] = (1 - probabilities) * 100
    st.markdown("### 📈 Financial Health Score Distribution")
    st.bar_chart(results.set_index("Company")["Financial Health Score (0-100)"])

    # ===============================
    # 💾 Download Section
    # ===============================
    st.markdown("<h3 style='margin-top:40px;'>💾 Download Your Results</h3>", unsafe_allow_html=True)

    # CSV Download
    csv_data = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV Report",
        data=csv_data,
        file_name="FinGuard_Predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ===============================
    # 📄 Generate PDF (No None Issue)
    # ===============================
    def generate_pdf(dataframe):
        pdf = FPDF()
        pdf.add_page()

        # Add logo
        if logo_path.exists():
            pdf.image(str(logo_path), x=80, y=10, w=50)
            pdf.ln(35)

        # Title & Summary
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 10, "FinGuard AI - Financial Health Report", ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Summary: {healthy} Healthy | {distressed} At Risk", ln=True, align="C")
        pdf.cell(0, 10, f"Average Health Score: {avg_health:.2f}%", ln=True, align="C")
        pdf.ln(10)

        # Table Header
        pdf.set_font("Arial", "B", 12)
        pdf.cell(60, 10, "Company", 1, 0, "C")
        pdf.cell(60, 10, "Status", 1, 0, "C")
        pdf.cell(60, 10, "Risk (%)", 1, 1, "C")

        # Table Rows
        pdf.set_font("Arial", "", 10)
        fill = False
        for _, row in dataframe.iterrows():
            pdf.set_fill_color(245, 245, 245) if fill else pdf.set_fill_color(255, 255, 255)
            fill = not fill
            pdf.set_text_color(0, 150, 0) if row["Predicted Status"] == "Healthy" else pdf.set_text_color(220, 0, 0)
            pdf.cell(60, 8, str(row["Company"]), 1, 0, "", fill)
            pdf.cell(60, 8, row["Predicted Status"], 1, 0, "", fill)
            pdf.cell(60, 8, str(row["Risk Probability (%)"]), 1, 1, "", fill)

        # Reset text color
        pdf.set_text_color(0, 0, 0)
        pdf.ln(10)

        # Footer
        pdf.set_font("Arial", "I", 10)
        pdf.multi_cell(
            0, 8,
            "Built by Ved Namde | FinGuard AI © 2025\nEmpowering Smarter Financial Decisions.",
            align="C"
        )

        # ✅ Correct way to get PDF bytes without showing "None"
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer, dest="F")  # Force output to buffer
        pdf_buffer.seek(0)
        return pdf_buffer.read()


        pdf_bytes = generate_pdf(results)
        st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="FinGuard_Report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


else:
    st.info("👆 Please upload a CSV file to start predictions.")

# ===============================
# 💬 Footer
# ===============================
st.markdown("""
<hr style='margin-top:50px; border:1px solid #333;'>
<div style='text-align:center; padding:10px; color:#aaa;'>
    Built with ❤️ by <strong>Ved Namde</strong> | <em>FinGuard AI © 2025</em><br>
    <span style='font-size:14px; color:#00e676;'>Empowering Smarter Financial Decisions.</span>
</div>
""", unsafe_allow_html=True)
