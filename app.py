import streamlit as st
from src.pipeline.predictionpipeline import Predicting


st.title("📧 Spam Prediction System")

email = st.text_area(
    "Enter the email:",
    placeholder="Paste your email here..."
)

if st.button("Predict"):
    if email:

        # Your model prediction
        prediction=Predicting()
        pred1=prediction.Pred(email)
        if pred1[0] == 1:
            st.error("🚨 Email is Spam")
        else:
            st.success("✅ Email is Not Spam")

    else:
        st.warning("Please enter an email.")


# if st.button("Predict"):
#     prediction=Predicting()
#     pred1=prediction.Pred(email)
#     if pred1[0]==1:
#         st.error("Email is spam")
# else:
#     st.success("Email is not spam")