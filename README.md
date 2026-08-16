# Driver Drowsiness Detection

A TensorFlow, OpenCV, and Streamlit project for research on driver-drowsiness detection.

## What the application does

- Detects the largest face in an image, uploaded video, or live webcam stream.
- Combines a MobileNetV2 face classifier with an independent open-eye check.
- Requires sustained evidence over time before raising a video/webcam alert; one high-scoring frame is not enough.
- Shows model score, rolling closed-eye ratio, alert timeline, annotated video, and downloadable evidence.
- Keeps static-image output conservative: two detected open eyes override a drowsy CNN score.
- Labels the CNN output as a model score, not a calibrated medical probability.

## Project structure

```text
.
|-- app.py                         # Streamlit application
|-- drowsiness_temporal.py         # Stateful video/webcam decision logic
|-- train_improved_model.py        # Subject-independent training/evaluation
|-- DDD_Drowsiness_Executed.ipynb  # Original coursework experiments
|-- run_local_notebook.py          # Original E1-E6 runner
|-- HUONG_DAN_CAI_DAT.md
`-- outputs/
    |-- models/
    |   |-- E6_MobileNetV2_subject.keras
    |   `-- improved_mobilenetv2.keras
    |-- results/improved_model_metadata.json
    `-- tables/improved_subject_metrics.csv
```

## Run the application Local

Use Python 3.11. Full setup instructions are in [HUONG_DAN_CAI_DAT.md](HUONG_DAN_CAI_DAT.md).

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Commit and push `app.py`, `drowsiness_temporal.py`, `requirements.txt`,
   `outputs/models/improved_mobilenetv2.keras`, and
   `outputs/results/improved_model_metadata.json` to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/), sign in with
   GitHub, and select **Create app**.
3. Choose this repository, branch `main`, and entry point `app.py`.
4. In advanced settings, select Python 3.11 when that choice is available.
5. Deploy. No secrets are required by this application.

The tracked model is the final 160×160 checkpoint only. Training-phase models,
datasets, notebooks outputs, and local virtual environments are intentionally
excluded from the deployment bundle.

## Train and evaluate the improved pipeline

First run the smoke test:

```powershell
.\.venv\Scripts\python.exe train_improved_model.py --quick
```

Then run all images and the two-stage training schedule:

```powershell
.\.venv\Scripts\python.exe train_improved_model.py
```

The split is grouped by subject, so a person cannot occur in more than one of train, validation, and test. The threshold is selected on validation data only. Final reporting includes image-level and subject-macro accuracy, balanced accuracy, precision, recall, F1, ROC AUC, and a confusion matrix.

Additional locally collected datasets can be included without mixing their subject identifiers with DDD:

```powershell
.\.venv\Scripts\python.exe train_improved_model.py --extra-data "D:\data\my_driver_faces"
```

Keep the same `Drowsy` and `Non-Drowsy` directory naming convention. Collect consented data under realistic camera, lighting, glasses, head-pose, and driver-demographic conditions, and assign stable subject folders/identifiers.

The app deploys `improved_mobilenetv2.keras` only when it came from a full run and achieved both test ROC AUC and balanced accuracy of at least 0.55 on unseen subjects. Otherwise it safely retains the included E6 fallback. A quick smoke-test model is never deployed.

## Important limitation

This remains an academic prototype. A face classifier cannot directly know whether someone is sleepy, and Haar eye detection is not a certified PERCLOS measurement. Validate on held-out drivers and real driving-like video before drawing conclusions. Do not use this project as a road-safety device, medical diagnostic system, or production alarm.
