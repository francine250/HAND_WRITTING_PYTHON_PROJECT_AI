# Handwriting → Computer File (AI Project)
Module ITLPA701 – Python and Fundamentals of AI · CAT 2 Practical

## 1. Problem domain
Many people still keep notes, letters and forms **handwritten**. Retyping is slow. This app takes an uploaded
handwritten image/PDF and converts it to an editable **TXT, Word (.docx) or PDF** file.
Domain: **image recognition + natural language processing** (handwritten text recognition).

## 2. AI approach – Deep Learning (pretrained)
**TrOCR (`microsoft/trocr-large-handwritten`)** from Hugging Face: a Vision Transformer encoder (reads the image)
+ Transformer decoder (writes the text). Pretrained on handwriting, so no training from scratch is needed.
*Why:* classic CNN-on-characters (EMNIST) cannot read connected cursive/words. The app defaults to the larger
TrOCR model and also offers the smaller base model. The larger model needs more memory and may run more slowly.
Both models read single text lines and can still make mistakes on handwriting unlike their training data.

## 3. Resources / dataset
- Environment: Python 3.10+, PyCharm / Jupyter / Spyder → `pip install -r requirements.txt`
- Data: **IAM Handwriting (Kaggle)** line images, or 10–20 pages written by the candidate/lecturer-approved.
  Put images in `dataset/images/NAME.png` and the correct typed text in `dataset/labels/NAME.txt`.
- Preprocessing (`preprocessing.py`): PDF→image, greyscale, denoise, lighting correction, adaptive threshold,
  deskew, **line segmentation** (horizontal projection profile) → each line is fed to the model.

## 4. Architecture
```
Upload (JPG/PNG/PDF)
   → Page loader (PyMuPDF/OpenCV)
   → Preprocessing (denoise, de-light, deskew)
   → Line segmentation (projection profile)
   → TrOCR  [ViT encoder → Transformer decoder, beam search]
   → Join lines → editable text box (human review)
   → Export: TXT / DOCX / PDF
```

## 5. Files
| File | Purpose |
|---|---|
| `preprocessing.py` | load, clean, deskew, segment lines |
| `ocr_engine.py` | TrOCR model, transcription pipeline, activity log, model saving |
| `exporter.py` | create TXT / DOCX / PDF |
| `evaluate.py` | CER / WER evaluation + parameter tuning + saves best config & model |
| `app.py` | Streamlit web interface |
| `config.json` | all settings (saved for reproducibility) |

## 6. Important parameters (explain in your defence)
| Parameter | Meaning | Effect |
|---|---|---|
| `num_beams` | beam-search width of decoder | higher = more accurate, slower |
| `max_new_tokens` | max characters/tokens per line | prevents endless output |
| `binarize` | convert to pure black/white | helps faint pens, can hurt thick ones |
| `line_proj_ratio` | ink threshold for detecting a text line | too low merges lines, too high misses faint ones |
| `merge_gap`, `min_line_height`, `line_padding` | line-segmentation tuning | avoids splitting letters like i/j or keeping noise |

## 7. Evaluation (metrics for the Deep Learning approach)
- **CER** (Character Error Rate) and **WER** (Word Error Rate) = edit-distance / reference length. Lower is better.
  Accuracy ≈ 1 − CER. Also **response time per page**.
- `python evaluate.py` (add `--single_line` for IAM line images) tests beams ∈ {1,3,5} × binarize ∈ {off,on},
  writes `results/tuning_results.csv`, `results/tuning_chart.png`, picks the best setting, updates `config.json`
  and saves the model to `saved_model/`.
- Interpretation example: *CER 0.08 means about 8 wrong characters per 100.*
  Typical TrOCR results on neat handwriting: CER 5–10 %; messy cursive: 15 %+.

## 8. Run / demonstrate
```
pip install -r requirements.txt
python evaluate.py            # once (needs internet the first time to download the model)
streamlit run app.py          # opens http://localhost:8501
```
The first use of a Hugging Face model requires an internet connection to download its files. For the best result, photograph one page at a time, flat and straight, in bright even light; keep the whole page in focus and avoid shadows. In the app, preview the detected line crops before converting and check that each handwritten row appears separately. If rows are combined, raise **Line detection sensitivity** or reduce **Join line fragments**; if faint rows disappear, lower sensitivity or **Minimum line height**. Increase **Maximum output tokens per line** if the end of a line is cut off. After conversion, enter the document title and correct the content, keeping each list item on its own line. The preview and TXT, Word, and PDF downloads use a clean title-and-bullets layout. OCR cannot guarantee an exact transcription, so compare it with the image and remove errors before exporting.
Activity is logged in `logs/activity.log`.

## 9. Responsible use
| Risk / limitation | How it is reduced |
|---|---|
| Misreads messy handwriting, numbers, names | Editable review box before export; show CER in report |
| Private documents uploaded | Runs locally, no cloud calls; warn users in the app |
| Bias – trained mostly on English handwriting | Test on local writers; fine-tune with Kinyarwanda samples if needed |
| Wrong line splitting on tilted/crowded pages | Deskew, adjustable parameters, "single line" mode |

## 10. Rubric mapping
Environment & functionalities → requirements.txt, §3 · Data acquired/pre-processed → `dataset/`, `preprocessing.py` ·
Features engineered → cleaning + line segmentation · Model selected/justified → §2 · Parameters → §6 ·
Implementation & testing → `ocr_engine.py`, `app.py` · Metrics/evaluation/tuning → `evaluate.py` ·
Model saved → `saved_model/`, `config.json` · Deployed → Streamlit.
