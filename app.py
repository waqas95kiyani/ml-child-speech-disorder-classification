# importing all the required libraries
import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd
import librosa
import torch
import joblib
import opensmile

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from transformers import Wav2Vec2Processor, Wav2Vec2Model


# page setup
st.set_page_config(
    page_title="Child Speech Screening Tool",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .result-card {
        padding: 28px;
        border-radius: 18px;
        margin-top: 15px;
        margin-bottom: 25px;
        font-size: 28px;
        font-weight: 800;
    }

    .td-card {
        background-color: rgba(34, 197, 94, 0.16);
        border: 1px solid rgba(34, 197, 94, 0.55);
        color: #4ADE80;
    }

    .ssd-card {
        background-color: rgba(239, 68, 68, 0.16);
        border: 1px solid rgba(239, 68, 68, 0.55);
        color: #F87171;
    }

    .vote-box {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .vote-title {
        color: #38BDF8;
        font-weight: 800;
        letter-spacing: 3px;
        font-size: 15px;
        margin-bottom: 22px;
    }

    .vote-row {
        display: flex;
        align-items: center;
        margin: 20px 0;
        gap: 14px;
    }

    .vote-label {
        width: 80px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .td-label {
        color: #34D399;
    }

    .ssd-label {
        color: #F87171;
    }

    .bar-bg {
        flex-grow: 1;
        background-color: #1F2937;
        height: 14px;
        border-radius: 12px;
        overflow: hidden;
    }

    .bar-fill-td {
        background-color: #34D399;
        height: 100%;
        border-radius: 12px;
    }

    .bar-fill-ssd {
        background-color: #F87171;
        height: 100%;
        border-radius: 12px;
    }

    .vote-count {
        width: 80px;
        text-align: right;
        color: #CBD5E1;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Child Speech Screening Tool")

st.markdown("""
This tool performs **Speech Sound Disorder (SSD)** and **Typically Developing (TD)** screening
and generates an acoustic speech profile relative to a TD reference group.
""")

st.caption(
    "Research prototype only. This tool is designed to support screening and interpretation, not clinical diagnosis."
)


# loading models and objects

@st.cache_resource
def load_wav2vec_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()
    return processor, model


@st.cache_resource
def load_classifier_and_scaler():
    classifier = joblib.load("wav2vec_classifier.joblib")
    scaler = joblib.load("wav2vec_scaler.joblib")
    return classifier, scaler


@st.cache_resource
def load_profile_objects():
    td_mean = joblib.load("td_mean_egemaps.joblib")
    td_std = joblib.load("td_std_egemaps.joblib")
    td_baselines = joblib.load("td_baselines.joblib")
    clean_dimensions = joblib.load("clean_dimensions.joblib")
    return td_mean, td_std, td_baselines, clean_dimensions


@st.cache_resource
def load_opensmile():
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )


processor, wav2vec_model = load_wav2vec_model()
classifier, scaler = load_classifier_and_scaler()
td_mean, td_std, td_baselines, clean_dimensions = load_profile_objects()
smile = load_opensmile()


# helper Functions
# for embeddings
def extract_wav2vec_embedding(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = wav2vec_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().numpy()

# for eGeMAP features
def extract_egemaps_features(audio_paths):
    rows = []

    for audio_path in audio_paths:
        features = smile.process_file(audio_path)
        rows.append(features)

    egemaps_df = pd.concat(rows, ignore_index=True)
    return egemaps_df

# for  profile scoring
def calculate_uploaded_child_profile(child_egemaps, td_mean, td_std, clean_dimensions):
    child_z_abs = ((child_egemaps - td_mean) / td_std).abs()

    test_profiles = {}

    for dimension, subgroups in clean_dimensions.items():
        subgroup_scores = {}

        for subgroup, features in subgroups.items():
            available_features = [
                f for f in features
                if f in child_z_abs.columns
            ]

            if len(available_features) > 0:
                subgroup_scores[subgroup] = child_z_abs[available_features].mean(axis=1)

        test_profiles[dimension] = pd.DataFrame(subgroup_scores)

    return test_profiles

# for speech profile plot
def plot_all_dimension_radars_plotly(child_label, td_baselines, test_profiles):

    dimensions = list(td_baselines.keys())
    n_dims = len(dimensions)

    n_cols = 3
    n_rows = int(np.ceil(n_dims / n_cols))

    specs = [[{"type": "polar"} for _ in range(n_cols)] for _ in range(n_rows)]
    subplot_titles = [f"<b>{d}</b>" for d in dimensions]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.28,
        vertical_spacing=0.28
    )

    for annotation in fig.layout.annotations:
        annotation.font = dict(size=16, color="white")
        annotation.y = annotation.y + 0.04

    td_color = "#38BDF8"
    child_color = "#F97316"

    for i, dimension in enumerate(dimensions):

        row = i // n_cols + 1
        col = i % n_cols + 1

        td_baseline = td_baselines[dimension]
        test_profile_df = test_profiles[dimension]

        labels = td_baseline.index.tolist()

        td_values = td_baseline.values.tolist()
        child_values = test_profile_df.iloc[0][labels].values.tolist()

        labels_closed = labels + [labels[0]]
        td_closed = td_values + [td_values[0]]
        child_closed = child_values + [child_values[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=td_closed,
                theta=labels_closed,
                mode="lines+markers",
                name="Average TD Reference",
                legendgroup="TD",
                showlegend=(i == 0),
                line=dict(color=td_color, width=3),
                marker=dict(color=td_color, size=7),
                fill="toself",
                fillcolor="rgba(56,189,248,0.18)"
            ),
            row=row,
            col=col
        )

        fig.add_trace(
            go.Scatterpolar(
                r=child_closed,
                theta=labels_closed,
                mode="lines+markers",
                name=child_label,
                legendgroup="Child",
                showlegend=(i == 0),
                line=dict(color=child_color, width=3),
                marker=dict(color=child_color, size=7),
                fill="toself",
                fillcolor="rgba(249,115,22,0.24)"
            ),
            row=row,
            col=col
        )

    all_values = []

    for dimension in dimensions:
        all_values.extend(td_baselines[dimension].values.tolist())
        all_values.extend(test_profiles[dimension].iloc[0].values.tolist())

    max_r = max(2.5, np.nanmax(all_values) + 0.3)

    polar_layout = dict(
        radialaxis=dict(
            range=[0, max_r],
            tickfont=dict(size=10, color="white"),
            gridcolor="rgba(255,255,255,0.25)",
            linecolor="rgba(255,255,255,0.35)",
            showline=True
        ),
        angularaxis=dict(
            tickfont=dict(size=11, color="white"),
            gridcolor="rgba(255,255,255,0.25)",
            linecolor="rgba(255,255,255,0.35)"
        ),
        bgcolor="#111827"
    )

    for i in range(1, n_dims + 1):
        polar_name = "polar" if i == 1 else f"polar{i}"
        fig.update_layout(**{polar_name: polar_layout})

    fig.update_layout(
        title=dict(
            text="<b>Acoustic Speech Profile</b><br>TD Reference vs Uploaded Child",
            x=0.5,
            y=0.98,
            xanchor="center",
            font=dict(size=22, color="white")
        ),
        height=1000,
        autosize=True,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white"),
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.10,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=13, color="white")
        ),
        margin=dict(t=180, b=110, l=80, r=80)
    )

    fig.add_annotation(
        text="<b>Scores = mean absolute z-score from TD baseline | closer to centre = TD-like | further outward = greater deviation</b>",
        x=0.5,
        y=-0.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=13, color="white")
    )

    return fig

# for prediction table at utterance level
def create_prediction_table(uploaded_audios, utterance_preds):
    return pd.DataFrame({
        "Audio File": [audio.name for audio in uploaded_audios],
        "Prediction": [
            "Typically Developing" if pred == 0 else "Possible SSD"
            for pred in utterance_preds
        ]
    })

# for vote breakdown
def vote_breakdown_html(td_votes, ssd_votes):
    total = max(td_votes + ssd_votes, 1)

    td_percent = (td_votes / total) * 100
    ssd_percent = (ssd_votes / total) * 100

    return f"""
<div class="vote-box">
<div class="vote-title">UTTERANCE VOTE BREAKDOWN</div>

<div class="vote-row">
<div class="vote-label td-label">TD</div>
<div class="bar-bg">
<div class="bar-fill-td" style="width:{td_percent}%;"></div>
</div>
<div class="vote-count">{td_votes}/{total}</div>
</div>

<div class="vote-row">
<div class="vote-label ssd-label">SSD</div>
<div class="bar-bg">
<div class="bar-fill-ssd" style="width:{ssd_percent}%;"></div>
</div>
<div class="vote-count">{ssd_votes}/{total}</div>
</div>
</div>
"""


# Upload section to allow files upload in the form of recordings

uploaded_audios = st.file_uploader(
    "Upload child speech audio files",
    type=["wav", "mp3", "m4a"],
    accept_multiple_files=True
)


# main App

if uploaded_audios:

    st.success(f"{len(uploaded_audios)} audio files uploaded successfully.")

    audio_paths = []

    for uploaded_audio in uploaded_audios:
        suffix = os.path.splitext(uploaded_audio.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_audio.read())
            audio_paths.append(tmp_file.name)

    tab_result, tab_profile, tab_audio, tab_details = st.tabs(
        [
            "Screening Result",
            "Acoustic Speech Profile",
            "Audio Preview",
            "Utterance Details"
        ]
    )

    all_embeddings = []

    with st.spinner("Extracting wav2vec2 embeddings and generating screening result..."):
        for audio_path in audio_paths:
            embedding = extract_wav2vec_embedding(audio_path)
            all_embeddings.append(embedding)

    all_embeddings = np.array(all_embeddings)
    all_embeddings_scaled = scaler.transform(all_embeddings)

    utterance_preds = classifier.predict(all_embeddings_scaled)

    prediction = pd.Series(utterance_preds).mode()[0]

    td_votes = int(np.sum(utterance_preds == 0))
    ssd_votes = int(np.sum(utterance_preds == 1))
    total_votes = len(utterance_preds)

    vote_confidence = max(td_votes, ssd_votes) / total_votes

    if prediction == 1:
        result_label = "Possible Speech Sound Disorder"
        result_icon = "❌"
        result_card_class = "ssd-card"
    else:
        result_label = "Typically Developing"
        result_icon = "✅"
        result_card_class = "td-card"

    # section 1: Screening Result

    with tab_result:

        st.subheader("Final Child-Level Screening Result")

        st.markdown(
            f"""
<div class="result-card {result_card_class}">
{result_icon} Prediction: {result_label}
</div>
""",
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("TD Votes", td_votes)
        col2.metric("SSD Votes", ssd_votes)
        col3.metric("Vote Confidence", f"{vote_confidence:.1%}")

        st.caption(
            "Child-level prediction is generated using majority voting across utterance-level predictions."
        )

    # section 2: Acoustic Speech Profile

    with tab_profile:

        st.subheader("Acoustic Speech Profile")

        with st.spinner("Extracting eGeMAPS features and generating profile..."):

            egemaps_df = extract_egemaps_features(audio_paths)

            child_egemaps = pd.DataFrame([egemaps_df.mean(axis=0)])

            uploaded_test_profiles = calculate_uploaded_child_profile(
                child_egemaps=child_egemaps,
                td_mean=td_mean,
                td_std=td_std,
                clean_dimensions=clean_dimensions
            )

            fig = plot_all_dimension_radars_plotly(
                child_label="Uploaded Child",
                td_baselines=td_baselines,
                test_profiles=uploaded_test_profiles
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"responsive": True}
            )

        st.info(
            "Speech profile scores represent average absolute z-score deviation from the TD baseline. "
            "Higher outward values indicate greater deviation from typical speech patterns."
        )

    # section 3: Audio Preview

    with tab_audio:

        st.subheader("Audio Preview")

        selected_idx = st.selectbox(
            "Select an audio file to preview",
            options=list(range(len(uploaded_audios))),
            format_func=lambda i: uploaded_audios[i].name
        )

        st.audio(audio_paths[selected_idx])

        st.caption(
            "Only one audio player is shown at a time to keep the interface clean for large uploads."
        )

    # section 4: Utterance Details

    with tab_details:

        st.markdown(
            vote_breakdown_html(td_votes, ssd_votes),
            unsafe_allow_html=True
        )
        st.subheader("Utterance-Level Predictions")

        prediction_table = create_prediction_table(uploaded_audios, utterance_preds)

        st.dataframe(
            prediction_table,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These are individual utterance-level predictions. "
            "The final child-level result is based on majority voting."
        )

    st.divider()

    st.caption(
        "The screening result and acoustic profile are research outputs only. "
        "They should not be interpreted as a clinical diagnosis."
    )

else:
    st.info("Upload one or more child speech audio files to begin screening.")