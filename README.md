# AI Defect Detection

EfficientNetB0 + GradCAM for steel surface defect classification (NEU dataset, 6 classes).

---

## Project Structure

```
defect_detection/
├── data/
│   ├── raw/                ← put NEU dataset here
│   └── processed/
│       ├── train/
│       └── val/
├── models/
│   └── best_model.h5
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── gradcam.py
│   └── predict.py
├── notebooks/
├── app.py
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle:  
https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

Extract into `data/raw/`:
```
data/raw/
├── Crazing/
├── Inclusion/
├── Patches/
├── Pitted_surface/
├── Rolled-in_scale/
└── Scratches/
```

Then run in order:
```bash
python src/prepare_data.py
python src/train.py
streamlit run app.py
```

---

## Dataset

| Class | Description | Images |
|---|---|---|
| Crazing | fine cracks | 300 |
| Inclusion | embedded foreign material | 300 |
| Patches | blotchy surface | 300 |
| Pitted Surface | small pits | 300 |
| Rolled-in Scale | scale pressed in during rolling | 300 |
| Scratches | scratch marks | 300 |

---

## Author
**Heart Khunpanuk** — Metrology & Integration Engineer
