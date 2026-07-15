import os
from datetime import datetime

import streamlit as st
import torch
import pandas as pd

from model import BiLSTMEmotionClassifier
from predict import predict_student_emotion
from predict_bert import load_bert_components, predict_student_emotion_bert
from vocab_builder import basic_tokenizer


# =====================================
# Configuration
# =====================================

st.set_page_config(
    page_title="Student Emotion Detection",
    page_icon="🧠",
    layout="wide"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HISTORY_FILE = "session_history.csv"

HISTORY_COLUMNS = [
    "timestamp",
    "input_text",
    "bilstm_emotion",
    "bilstm_confidence",
    "bert_emotion",
    "bert_confidence",
    "agreement",
    "final_emotion",
]


# =====================================
# History Helpers
# =====================================

def log_prediction(
    input_text,
    bilstm_emotion,
    bilstm_confidence,
    bert_emotion,
    bert_confidence,
    final_emotion
):

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_text": input_text,
        "bilstm_emotion": bilstm_emotion,
        "bilstm_confidence": round(float(bilstm_confidence) * 100, 2),
        "bert_emotion": bert_emotion,
        "bert_confidence": round(float(bert_confidence) * 100, 2),
        "agreement": bilstm_emotion == bert_emotion,
        "final_emotion": final_emotion,
    }

    df_new = pd.DataFrame([entry], columns=HISTORY_COLUMNS)

    if os.path.exists(HISTORY_FILE):
        df_new.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df_new.to_csv(HISTORY_FILE, mode="w", header=True, index=False)


def load_history():

    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    df = pd.read_csv(HISTORY_FILE)

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def clear_history():

    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


# =====================================
# Load BiLSTM Model (weights + vocab + config all come from one checkpoint)
# =====================================

@st.cache_resource
def load_bilstm():

    checkpoint = torch.load(
        "bilstm_emotion_model.pt",
        map_location=DEVICE
    )

    vocab = checkpoint["vocab"]
    config = checkpoint["config"]

    model = BiLSTMEmotionClassifier(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        dropout_prob=config.get("dropout_prob", 0.3)
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model, vocab


model, vocab = load_bilstm()


# =====================================
# Load BERT Model (cached, not loaded at import time)
# =====================================

@st.cache_resource
def load_bert():
    return load_bert_components()


bert_tokenizer, bert_model, bert_label_classes = load_bert()


# =====================================
# Emotion Settings
# Keys MUST match EMOTION_LABELS in predict.py / EMOTION_MAP in train_bert.py
# =====================================

EMOTION_STYLE = {

    "Bored": {"emoji": "😐", "color": "#95a5a6"},
    "Confident": {"emoji": "😊", "color": "#2ecc71"},
    "Confused": {"emoji": "😕", "color": "#3498db"},
    "Curious": {"emoji": "🤔", "color": "#f39c12"},
    "Frustrated": {"emoji": "😡", "color": "#e74c3c"}

}


GUIDANCE_TEXT = {

    "Bored": [
        "Try a short break, then re-engage.",
        "Mix in a hands-on exercise or example."
    ],

    "Confident": [
        "Great engagement!",
        "Keep maintaining your positive attitude."
    ],

    "Confused": [
        "Take small breaks.",
        "Ask for a worked example or revisit the basics."
    ],

    "Curious": [
        "Follow that curiosity — explore further!",
        "Look up a related topic to go deeper."
    ],

    "Frustrated": [
        "Try relaxation techniques.",
        "Take deep breaths, then retry step by step."
    ]

}


# =====================================
# Sidebar Navigation
# =====================================

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Predict Emotion", "📊 Analytics", "🕒 Session History"]
)


# =====================================
# PAGE 1: Predict Emotion
# =====================================

def render_predict_page():

    st.title(
        "🧠 Student Emotion Detection using BiLSTM + BERT"
    )

    user_input = st.text_area(
        "Enter student text:"
    )

    predict_button = st.button(
        "Predict Emotion"
    )

    if predict_button:

        if not user_input.strip():

            st.warning(
                "Please enter some text."
            )

        else:

            try:

                # -------------------------
                # BiLSTM Prediction
                # -------------------------

                bilstm_result = predict_student_emotion(
                    model,
                    basic_tokenizer,
                    vocab,
                    user_input,
                    DEVICE
                )

                emotion = bilstm_result["predicted_emotion"]
                confidence = bilstm_result["confidence"]

                # -------------------------
                # BERT Prediction
                # -------------------------

                bert_result = predict_student_emotion_bert(
                    user_input,
                    bert_tokenizer,
                    bert_model,
                    bert_label_classes
                )

                bert_emotion = bert_result["predicted_emotion"]
                bert_confidence = bert_result["confidence"]

                # -------------------------
                # Display Results
                # -------------------------

                st.subheader(
                    "🧠 Model Predictions"
                )

                col1, col2 = st.columns(2)

                # BiLSTM Card

                with col1:

                    style = EMOTION_STYLE.get(
                        emotion,
                        {"emoji": "🙂", "color": "#4169e1"}
                    )

                    st.markdown(
                        f"""
                        <div style="
                        border-left:6px solid {style['color']};
                        padding:15px;
                        ">

                        <h2>🤖 BiLSTM</h2>

                        <h3>{style['emoji']} {emotion}</h3>

                        <p><b>{emotion}</b>: {confidence*100:.2f}%</p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # BERT Card

                with col2:

                    style = EMOTION_STYLE.get(
                        bert_emotion,
                        {"emoji": "🙂", "color": "#4169e1"}
                    )

                    st.markdown(
                        f"""
                        <div style="
                        border-left:6px solid {style['color']};
                        padding:15px;
                        ">

                        <h2>🧠 BERT</h2>

                        <h3>{style['emoji']} {bert_emotion}</h3>

                        <p><b>{bert_emotion}</b>: {bert_confidence*100:.2f}%</p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # -------------------------
                # Comparison
                # -------------------------

                st.subheader(
                    "⚖️ Model Comparison"
                )

                if emotion == bert_emotion:

                    st.success(
                        f"""
                        Both models agree ✅

                        Final Emotion:
                        **{emotion}**
                        """
                    )

                else:

                    st.warning(
                        f"""
                        Models disagree ⚠️

                        BiLSTM:
                        **{emotion}**

                        BERT:
                        **{bert_emotion}**
                        """
                    )

                # -------------------------
                # Confidence Table
                # -------------------------

                st.subheader(
                    "📊 Confidence Comparison"
                )

                st.table(
                    {
                        "Model": ["BiLSTM", "BERT"],
                        "Emotion": [emotion, bert_emotion],
                        "Confidence": [
                            f"{confidence*100:.2f}%",
                            f"{bert_confidence*100:.2f}%"
                        ]
                    }
                )

                # -------------------------
                # Guidance
                # -------------------------

                st.subheader(
                    "💡 Personalized Guidance"
                )

                final_emotion = (
                    bert_emotion
                    if bert_confidence > confidence
                    else emotion
                )

                for tip in GUIDANCE_TEXT.get(
                    final_emotion,
                    ["Keep learning."]
                ):

                    st.write("✅", tip)

                # -------------------------
                # Log to Session History
                # -------------------------

                log_prediction(
                    input_text=user_input,
                    bilstm_emotion=emotion,
                    bilstm_confidence=confidence,
                    bert_emotion=bert_emotion,
                    bert_confidence=bert_confidence,
                    final_emotion=final_emotion
                )

                st.caption(
                    "📝 Saved to Session History — check the sidebar to view it."
                )

            except Exception as e:

                st.exception(e)


# =====================================
# PAGE 2: Analytics
# =====================================

def render_analytics_page():

    st.title("📊 Analytics Dashboard")

    df = load_history()

    if df.empty:
        st.info("No predictions yet.")
        return

    latest = df.iloc[-1]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "BiLSTM",
        f"{latest['bilstm_confidence']:.2f}%"
    )

    col2.metric(
        "BERT",
        f"{latest['bert_confidence']:.2f}%"
    )

    col3.metric(
        "Final Emotion",
        latest["final_emotion"]
    )

    st.divider()

    st.subheader("📊 Latest Prediction")

    bilstm_data = pd.DataFrame(
        {
            "Emotion": [latest["bilstm_emotion"]],
            "Confidence": [latest["bilstm_confidence"]]
        }
    )

    bert_data = pd.DataFrame(
        {
            "Emotion": [latest["bert_emotion"]],
            "Confidence": [latest["bert_confidence"]]
        }
    )

    final_data = pd.DataFrame(
        {
            "Emotion": [latest["final_emotion"]],
            "Confidence": [100]
        }
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🤖 BiLSTM")
        st.bar_chart(
            bilstm_data.set_index("Emotion")
        )

    with col2:
        st.markdown("### 🧠 BERT")
        st.bar_chart(
            bert_data.set_index("Emotion")
        )

    with col3:
        st.markdown("### ✅ Final Emotion")
        st.bar_chart(
            final_data.set_index("Emotion")
        )

# =====================================
# PAGE 3: Session History
# =====================================

def render_history_page():

    st.title("🕒 Session History")
    st.caption("Every prediction made in this app, most recent first.")

    df = load_history()

    if df.empty:
        st.info(
            "No predictions yet. Go to **🏠 Predict Emotion** and run one "
            "to start building your history."
        )
        return

    df = df.sort_values("timestamp", ascending=False)

    st.subheader("🔍 Filters")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        emotion_options = sorted(df["final_emotion"].unique().tolist())

        selected_emotions = st.multiselect(
            "Filter by final emotion",
            options=emotion_options,
            default=emotion_options
        )

    with filter_col2:
        search_text = st.text_input(
            "Search within input text",
            placeholder="e.g. exam, confused, project..."
        )

    filtered_df = df[df["final_emotion"].isin(selected_emotions)]

    if search_text.strip():
        filtered_df = filtered_df[
            filtered_df["input_text"].str.contains(
                search_text, case=False, na=False
            )
        ]

    st.caption(f"Showing {len(filtered_df)} of {len(df)} total records.")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": "Time",
            "input_text": "Student Text",
            "bilstm_emotion": "BiLSTM Emotion",
            "bilstm_confidence": st.column_config.NumberColumn(
                "BiLSTM Conf. (%)", format="%.2f"
            ),
            "bert_emotion": "BERT Emotion",
            "bert_confidence": st.column_config.NumberColumn(
                "BERT Conf. (%)", format="%.2f"
            ),
            "agreement": "Models Agreed?",
            "final_emotion": "Final Emotion",
        }
    )

    st.divider()

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        st.download_button(
            "⬇️ Download filtered history as CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="session_history.csv",
            mime="text/csv"
        )

    with action_col2:
        confirm_clear = st.checkbox("I understand this will delete all history")

        if st.button("🗑️ Clear all history", disabled=not confirm_clear):
            clear_history()
            st.success("History cleared. Refresh the page to see the change.")
            st.rerun()


# =====================================
# Router
# =====================================

if page == "🏠 Predict Emotion":
    render_predict_page()
elif page == "📊 Analytics":
    render_analytics_page()
elif page == "🕒 Session History":
    render_history_page()