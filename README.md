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