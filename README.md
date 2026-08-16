# Driver Drowsiness Detection

A deep learning-based driver drowsiness detection project using MobileNetV2, TensorFlow, OpenCV, and Streamlit.

## Features

- Detects the largest face in webcam, image, or video input.
- Predicts a drowsiness probability with a MobileNetV2 binary classifier.
- Provides real-time webcam inference through WebRTC.
- Shows the original and probability-annotated videos side by side for direct comparison.
- Extracts threshold-crossing alert frames from uploaded videos as separate JPEG images.
- Displays a probability timeline and exports alert JPEGs with a CSV manifest in one ZIP archive.
- Includes a portable local training runner for the DDD Kaggle dataset.
- Exports experiment metrics, tables, plots, and Keras models.

## Project structure

```text
.
├── app.py
├── DDD_Drowsiness_Executed.ipynb
├── run_local_notebook.py
├── HUONG_DAN_CAI_DAT.md
└── outputs/
    ├── figures/
    ├── models/E6_MobileNetV2_subject.keras
    ├── results/
    └── tables/
```

## Quick start

Use Python 3.11 and create a fresh virtual environment. Full Windows, macOS, Linux, training, and troubleshooting instructions are available in [HUONG_DAN_CAI_DAT.md](HUONG_DAN_CAI_DAT.md).

Run the Streamlit application on Windows:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Run a quick pilot training pass:

```powershell
.\.venv\Scripts\python.exe run_local_notebook.py --force
```

Run full training:

```powershell
.\.venv\Scripts\python.exe run_local_notebook.py --full --force
```

## Model status

The included `E6_MobileNetV2_subject.keras` model was produced by the E6 MobileNetV2 subject-wise experiment in `DDD_Drowsiness_Executed.ipynb`. It is currently a pilot/CPU artifact intended for coursework demonstration. Run full training before using the model for serious evaluation.

## Disclaimer

This project is an academic demonstration. It is not a certified road-safety device, medical diagnostic system, or production safety system.
