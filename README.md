# Childhood Speech Sound Disorder Screening using Machine Learning

## Overview

This project focuses on developing a machine learning-based screening framework for childhood Speech Sound Disorder (SSD) using child speech recordings from the UltraSuite dataset. The aim of the project was to classify child speech as either typically developing (TD) or speech-disordered (SSD), while also providing an interpretable acoustic speech profile to support understanding of the model output.

Traditional SSD assessment relies heavily on speech pathologists’ listening, transcription, and clinical judgement. While expert assessment remains essential, this project explores how machine learning can support early screening by analysing child speech recordings and providing a structured prediction with an acoustic summary.

This project was completed as part of the COMP6019 Master of Predictive Analytics majoring in Data Science final semester project at Curtin University.

## Project Aim

The main aim of this project was to develop and evaluate a machine learning-based screening system for childhood Speech Sound Disorder using child-only speech recordings from the UltraSuite dataset, supported by an interpretable eGeMAPS-based speech profile and a Streamlit demonstration application.

## Project Objectives

- Construct a usable research dataset from the UltraSuite repository using UXTD as the typically developing class and baseline UPX/UXSSD recordings as the speech-disordered class.
- Isolate child-only speech from raw therapy recordings using speaker labels and timestamps.
- Extract handcrafted eGeMAPS acoustic features and pretrained wav2vec 2.0 speech embeddings from processed child-only audio.
- Train and compare speaker-independent machine learning models for binary classification of TD and SSD speech.
- Evaluate model performance using screening-relevant metrics, including accuracy, SSD recall, and unweighted average recall.
- Develop an interpretable acoustic speech profile using eGeMAPS features.
- Build a Streamlit application prototype to demonstrate the complete screening pipeline.

## Dataset

This project used the publicly available UltraSuite speech dataset, which contains child speech therapy recordings and related annotation files.

The project used three UltraSuite subsets:

- **UXTD**: Typically developing child speech
- **UXSSD**: Speech-disordered child speech
- **UPX**: Speech-disordered child speech

Only waveform audio files, text annotations, and speaker-label files were used. Ultrasound files and other annotation layers were excluded because the project focused on audio-based SSD screening.

Raw audio files are not included in this repository due to dataset size and usage considerations.

## Data Preprocessing

The raw UltraSuite recordings contained both child and therapist speech. Therefore, one of the most important preprocessing steps was to isolate child-only speech before feature extraction and modelling.

The preprocessing pipeline included:

1. Reading raw audio files, text annotations, and speaker-label files.
2. Constructing metadata tables for UXTD, UXSSD, and UPX.
3. Using speaker labels to retain only segments marked as `CHILD`.
4. Removing therapist speech segments marked as `SLT`.
5. Trimming segment boundaries to reduce speaker overlap and noise.
6. Removing very short segments that were unlikely to contain useful speech.
7. Saving processed child-only audio files.
8. Creating a combined master metadata table for modelling.

## Feature Extraction

Two feature extraction workflows were used in this project.

### eGeMAPS Acoustic Features

The eGeMAPS feature set was extracted using openSMILE. This produced 88 handcrafted acoustic features per recording.

These features were used for:

- Baseline machine learning models
- Acoustic feature analysis
- Interpretable speech profile development

The eGeMAPS features were useful because they represent meaningful speech characteristics such as pitch, loudness, spectral balance, voice quality, formant structure, and temporal information.

### wav2vec 2.0 Embeddings

The second workflow used pretrained wav2vec 2.0 embeddings. The `facebook/wav2vec2-base-960h` model was used to extract deep speech representations from child-only audio recordings.

Each audio recording was converted into a fixed-length 768-dimensional embedding using mean pooling. These embeddings were then used as input features for the final classification model.

The wav2vec 2.0 embeddings provided stronger predictive performance, while eGeMAPS features were retained for interpretability.

## Models Tested

The following models were tested during the project:

- Support Vector Machine using eGeMAPS features
- Support Vector Machine with SMOTE using eGeMAPS features
- CatBoost using eGeMAPS features
- Multi-Layer Perceptron using eGeMAPS features
- Support Vector Machine using wav2vec 2.0 embeddings

A speaker-independent evaluation approach was used to reduce data leakage. This ensured that recordings from the same child were not split across training and testing data.

## Evaluation Strategy

The project used speaker-level evaluation because each child had multiple utterances. Instead of only evaluating individual recordings, predictions were aggregated at the child level using majority voting.

The main evaluation metrics were:

- Accuracy
- Recall for SSD
- Unweighted Average Recall

Recall and UAR were especially important because the project was designed as a screening-support tool. In a screening context, missing a possible SSD case is more serious than incorrectly flagging a typically developing child for further review.

## Results and Significance

The best-performing model was the **wav2vec 2.0 embeddings + Support Vector Machine (SVM)** model. This model achieved the strongest overall performance across both validation and held-out testing.

### Validation Results

During child-level Leave-One-Speaker-Out validation, the wav2vec 2.0 + SVM model achieved:

- **96% accuracy**
- **100% unweighted average recall (UAR)**
- **97% recall for SSD**

Among the eGeMAPS-based models, the standard SVM with eGeMAPS features also performed strongly, achieving:

- **94% accuracy**
- **95% UAR**
- **95% SSD recall**

### Held-Out Test Results

On the separate held-out test set of **16 unseen children**, the wav2vec 2.0 + SVM model achieved the best child-level performance:

- **94% accuracy**
- **88% recall**
- **94% unweighted average recall (UAR)**

The model correctly classified **15 out of 16 children** at the child level. It correctly identified **7 out of 8 SSD children** and correctly classified **all typically developing children** in the held-out test set.

### Significance of Results

These results are significant because the model was evaluated using a **speaker-independent approach**, meaning that children in the test set were not seen during model training. This reduced the risk of data leakage and provided a more realistic estimate of performance on unseen children.

The results also showed that **wav2vec 2.0 embeddings captured stronger speech representations** than handcrafted acoustic features for classification. However, eGeMAPS features remained important because they supported the development of an interpretable acoustic speech profile.

Overall, the project demonstrates that machine learning can support childhood speech disorder screening by combining:

- Strong predictive performance
- Child-level evaluation
- Interpretable acoustic profiling
- A practical Streamlit-based application prototype

## Acoustic Speech Profile

An interpretable acoustic speech profile was developed using eGeMAPS features. The profile was designed to provide an objective visual summary of a child’s speech characteristics compared with a typically developing reference group.

The speech profile included six major acoustic dimensions:

- Pitch / Prosody
- Loudness / Energy
- Voice Quality
- Spectral / Phonation
- Fluency / Rhythm
- Articulation / Vowel Quality

The profile was not designed to diagnose SSD directly. Instead, it was included as a support tool to help users understand which acoustic areas showed greater deviation from the TD reference group.

## Streamlit Application

A Streamlit application prototype was developed to demonstrate the practical use of the complete pipeline.

The application allows users to upload one or more child speech audio files and view:

- Final screening prediction
- Prediction confidence
- Acoustic speech profile
- Utterance-level prediction details
- Audio playback preview

The application is intended for demonstration and research purposes only and is not designed for clinical deployment.

## Screenshots

### Streamlit App Interface
![Streamlit App Interface](images/01-basic%20app%20interface.png)

### Screening Result
![Screening Result](images/02-screening%20result.png)

### Audio Review
![Audio Review](images/03-audio%20review.png)

### Utterance Vote Breakdown
![Utterance Vote Breakdown](images/04-utterance%20vote%20breakdown.png)

### Acoustic Speech Profile
![Acoustic Speech Profile](images/05-speech%20profile.png)

### Best Model Confusion Matrix
![Best Model Confusion Matrix](images/06-best%20model%20confusion%20matrix.png)

### LOSO Validation Results Across Models
![LOSO Validation Results Across Models](images/07-LOSO%20validation%20results%20(all%20models).png)

### Final Held-Out Test Results Across Models
![Final Held-Out Test Results Across Models](images/08-final%20held%20out%20test%20results%20(all%20models).png)

## Repository Contents

This repository includes:

- Metadata preparation notebooks
- Final modelling notebook
- Streamlit application file
- Sample metadata CSV files
- README documentation

Large generated files, such as full wav2vec 2.0 embeddings and trained model files, are not uploaded directly to this repository due to file size limitations.

## Large Files

Large generated feature files, including the full wav2vec 2.0 embeddings CSV, are not uploaded directly to this repository due to file size limitations.

The files are available here:

[Download large feature files in csv from Google Drive](https://drive.google.com/drive/folders/1b0xOoeCAfuvdx8_zl7M6oMR1QHbl9NkX?usp=sharing)

Included files:

- wav2vec 2.0 embeddings CSV
- extracted acoustic feature files
- model input feature files

## Important Note

This project is intended as a machine learning-based screening support tool. It is not a diagnostic system and should not replace assessment by qualified speech-language professionals.

# Instructions to Run the Code
The following code was tested on two different devices with the following configurations:

1. Primary Device  
   MacBook M4 Pro with 24 GB RAM
   OS Version: macOS Tahoe 26.4.1

2. Secondary Device  
   ASUS G14 Windows-based machine with 32 GB RAM and 1 TB storage
   OS Version: Windows 11 Version 25H2 (OS Build 26200.8457)

3. All the code development was done on Microsoft Visual Code with following extensions installed:
    Jupyter Notebook

4. This code performs segmentation on a large unstructured audio dataset. Therefore, make sure to have at least 5–10 GB of free space in the project directory, as the code stores cleaned audio files during the data cleaning and segmentation process.

5. If the user does not want to download the full UltraSuite dataset, it is recommended to run `final_code.ipynb` directly. This notebook uses the stored pre-extracted results from the CSV files to demonstrate the working of the complete pipeline.

### Note: 
- For maximum compatibility, it is recommended to test this code on a MacBook using VS Code.

# 1. Setting Up the Working Environment
Run the following commands one by one in the VS Code terminal.

## For Mac
```text
brew install python@3.11
python3.11 -m venv final_project_audio
source final_project_audio/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers librosa pandas numpy scikit-learn matplotlib seaborn jupyter ipykernel
pip install opensmile imbalanced-learn catboost plotly joblib streamlit
python -m ipykernel install --user --name final_project_audio --display-name "Python 3.11 Mac - Final Project Audio"
```
## For Windows 11
Open the project folder in **VS Code** and run the following commands in the **VS Code PowerShell terminal**.
```text
python -m venv final_project_audio
.\final_project_audio\Scripts\Activate.ps1
```

In case if some permission error occurs, use the following command:
```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

and then activate the environment using following command:
```
.\final_project_audio\Scripts\Activate.ps1
```
After the enviroment is activated, run the following commands:
```
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers librosa pandas numpy scikit-learn matplotlib seaborn jupyter ipykernel
pip install opensmile imbalanced-learn catboost plotly joblib streamlit
python -m ipykernel install --user --name final_project_audio --display-name "Python Windows - Final Project Audio"
```

# 2. Data Structure for Audio Preprocessing
Make sure that the UltraSuite dataset follows the directory structure below before running the code (inside the submitted zip folder), since all the data is not possible to upload. So the user needs to download the data from UltraSuite repo. Rest of the files in the directory will be provided in the uploaded zip folder.

```text
kiyani_21978638/
│
├── core_upx/
│   │
│   ├── core/
│   │   ├── 01F/
│   │   ├── 02F/
│   │   ├── 03F/
│   │   └── ...
│   │
│   └── speaker_labels/
│       ├── lab/
│       └── TG/
│
├── core_uxssd/
│   │
│   ├── core/
│   │   ├── 01M/
│   │   ├── 02M/
│   │   ├── 03F/
│   │   └── ...
│   │
│   └── speaker_labels/
│       ├── lab/
│       └── TG/
│
├── core_uxtd/
│   │
│   ├── core/
│   │   ├── 01M/
│   │   ├── 02M/
│   │   ├── 03F/
│   │   └── ...
│   │
│   └── speaker_labels/
│       ├── lab/
│       └── TG/
│
├── UXTD_info.csv
├── uxtd_metadata_clean.csv
├── UXSSD_info.csv
├── uxssd_metadata_clean.csv
├── UPX_info.csv
├── upx_metadata_clean.csv
├── df_embedding_extracted.csv
├── egemaps_features.csv
├── master_metadata.csv
├── app.py
├── 01-UPX_metadata.ipynb
├── 02-UXSSD_metadata.ipynb
├── 03-UXTD_metadata.ipynb
├── 04-final_code.ipynb
└── README.md
```

# 3. Running the Notebook Files

Run the files in the following order in VS Code using the environment kernel created above:

1. `UPX_metadata.ipynb`  
   Requires the UPX dataset to be downloaded.

2. `UXSSD_metadata.ipynb`  
   Requires the UXSSD dataset to be downloaded.

3. `UXTD_metadata.ipynb`  
   Requires the UXTD dataset to be downloaded.

4. `final_code.ipynb`  
   Can be run directly using the pre-extracted files which are already included in the project folder.

## Note:
The user can test the main output by running `final_code.ipynb` directly (no need to run the file 1,2 and 3) and then following Section 5 to test the Streamlit application.
This avoids the need to download the full UltraSuite dataset, as the required pre-extracted csv files are already included in the project folder.

# 4. wav2vec2 Embeddings and eGeMAPS Features

The final_code.ipynb file contains helper functions to extract both wav2vec2 embeddings and eGeMAPS features.

However, extracting all embeddings can take approximately 25–45 minutes on the tested systems. Therefore, the embedding extraction code has been commented out by default. The final code will use the pre-extracted wav2vec 2.0 embeddings already stored in the zip directory:
```
df_embedding_extracted.csv
```
Similarly, the eGeMAPS feature extraction code has also been commented out to avoid repeated processing time. The final code will use the pre-extracted eGeMAPS features stored in zip directory:
```
egemaps_features.csv
```
### Note

- If the user wants to test the feature extraction functions and repeat the full extraction process, they need to uncomment the relevant extraction function code in `final_code.ipynb`.

- At the same time, the user should comment out the existing code lines that read the pre-extracted files, such as `df_embeddings.csv` and `egemaps_features.csv`.

- The required instructions are also mentioned in the comments inside `final_code.ipynb`.

# 5. Profiling Test Section

To generate a speech profile for a test child, change the child_id value in the following code in the file `final_code.ipynb`:
```
plot_all_dimension_radars_plotly(
    child_id="upx_02F",  # Change the speaker ID to generate the profile for another child
    td_baselines=td_baselines,
    test_profiles=test_profiles)
```

# 6. Application Test

Run the following command in the terminal to launch the Streamlit application:
```
streamlit run app.py
```
The application will open in a browser. The user needs to upload multiple audio recordings in the upload section for the application to generate the output.

### Note: 
To test the app, the user must have one of the downloaded UltraSuite datasets: UPX, UXTD, or UXSSD.
