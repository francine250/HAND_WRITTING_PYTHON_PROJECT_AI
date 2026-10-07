"""Step 7 - Streamlit web application.  Run:  streamlit run app.py"""
from pathlib import Path

import streamlit as st
import cv2
from ocr_engine import HandwritingOCR, load_config
from preprocessing import load_pages, preprocess, segment_lines
from exporter import to_professional_docx, to_professional_pdf, to_professional_txt

st.set_page_config(page_title="Handwriting to Text", page_icon="✍️", layout="wide")
st.title("✍️ Handwriting → Computer File")
st.caption("Upload a clear, well-lit handwritten page. Results are a draft: review the text carefully before exporting.")

project_dir = Path(__file__).resolve().parent
cfg = load_config(str(project_dir / "config.json"))
saved_model_dir = Path(cfg["saved_model_dir"])
if not saved_model_dir.is_absolute():
    saved_model_dir = project_dir / saved_model_dir
cfg["saved_model_dir"] = str(saved_model_dir)

with st.sidebar:
    st.header("Settings")
    model_options = {
        "microsoft/trocr-large-handwritten": "Larger handwritten model (slower, more memory)",
        "microsoft/trocr-base-handwritten": "Smaller handwritten model (faster)",
    }
    model_names = list(model_options)
    configured_model = cfg.get("model_name", model_names[0])
    model_index = model_names.index(configured_model) if configured_model in model_names else 0
    cfg["model_name"] = st.selectbox(
        "Handwriting recognition model",
        model_names,
        index=model_index,
        format_func=lambda name: model_options[name],
        help="The larger model may improve recognition, but it is slower and needs more memory. Its first use downloads model files.",
    )
    if saved_model_dir.is_dir():
        cfg["use_saved_model"] = st.checkbox(
            "Use a locally saved model instead",
            bool(cfg.get("use_saved_model", False)),
            help="Uses the model in the saved_model folder rather than the selected Hugging Face model.",
        )
    else:
        cfg["use_saved_model"] = False
        st.info(
            "No saved model was found. The selected Hugging Face model will be used "
            "instead and may download files the first time it runs."
        )
    cfg["num_beams"] = st.slider("Beam search width (accuracy vs speed)", 1, 8, int(cfg["num_beams"]))
    cfg["binarize"] = st.checkbox("Black & white clean-up", cfg["binarize"])
    cfg["deskew"] = st.checkbox("Straighten tilted pages", cfg["deskew"])
    cfg["max_new_tokens"] = st.slider(
        "Maximum output tokens per line", 32, 256, int(cfg["max_new_tokens"]), 32,
        help="Increase this if a detected line seems cut off. Longer lines take more time to process.",
    )
    cfg["line_proj_ratio"] = st.slider(
        "Line detection sensitivity", 0.01, 0.35, float(cfg["line_proj_ratio"]), 0.01,
        help="Raise this if separate handwritten rows are combined; lower it if faint rows disappear.",
    )
    cfg["min_line_height"] = st.slider(
        "Minimum line height (pixels)", 8, 60, int(cfg["min_line_height"]), 2,
        help="Lower this if short handwriting lines are missing.",
    )
    cfg["merge_gap"] = st.slider(
        "Join line fragments (pixels)", 0, 30, int(cfg["merge_gap"]), 1,
        help="Increase this if one handwritten line is split into multiple detected crops.",
    )
    single = st.checkbox("The image contains one text line only", False)
    preview_lines = st.checkbox("Preview detected text lines before converting", True)
    show_lines = st.checkbox("Show detected lines", False)


@st.cache_resource(show_spinner="Loading AI model (first time only)...")
def get_ocr(model_name, saved_dir, use_saved_model):
    return HandwritingOCR({
        "model_name": model_name,
        "saved_model_dir": saved_dir,
        "use_saved_model": use_saved_model,
    })


ocr = get_ocr(cfg["model_name"], cfg["saved_model_dir"], cfg["use_saved_model"])
ocr.cfg = cfg

up = st.file_uploader("Upload handwritten file", type=["png", "jpg", "jpeg", "pdf"])
if up:
    try:
        pages = load_pages(up.getvalue(), up.name)
    except Exception as e:
        st.error(f"Could not read the file: {e}"); st.stop()

    if preview_lines:
        for page_number, page in enumerate(pages, 1):
            preview_gray, preview_ink = preprocess(page, cfg)
            crops = [preview_gray] if single else segment_lines(preview_gray, preview_ink, cfg)
            with st.expander(
                f"Detected lines — page {page_number} ({len(crops)} found)",
                expanded=True,
            ):
                if not crops:
                    st.warning("No text lines were detected. Try lowering line detection sensitivity.")
                for line_number, crop in enumerate(crops, 1):
                    st.image(
                        crop,
                        caption=f"Page {page_number}, line {line_number}",
                        width="stretch",
                        alt=f"Detected handwriting line {line_number} on page {page_number}",
                    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        for i, p in enumerate(pages, 1):
            st.image(
                cv2.cvtColor(p, cv2.COLOR_BGR2RGB),
                caption=f"Page {i}",
                width="stretch",
                alt=f"Uploaded handwritten page {i}",
            )

    if st.button("Convert to typed text", type="primary"):
        texts, total, nlines = [], 0.0, 0
        with st.spinner("Recognising handwriting..."):
            for p in pages:
                r = ocr.transcribe_page(p, single_line=single)
                texts.append(r["text"]); total += r["seconds"]; nlines += len(r["lines"])
                if show_lines:
                    for li in r["line_images"]:
                        st.image(li, width="content", alt="Detected handwriting line")
        st.session_state["text"] = "\n\n".join(texts)
        st.session_state["stats"] = (len(pages), nlines, total)
        st.session_state["document_title"] = ""
        st.session_state["document_items"] = st.session_state["text"]

    with col2:
        st.subheader("Professional typed document")
        if "text" in st.session_state:
            n, l, t = st.session_state["stats"]
            st.caption(f"{n} page(s) · {l} detected line(s) · {t:.1f}s")
            st.warning(
                "OCR is a draft and may contain missing or incorrect words. Correct the title "
                "and list items below, and remove anything that is not written in the image."
            )
            with st.expander("View raw OCR draft"):
                st.text(st.session_state["text"])
            title = st.text_input(
                "Document title",
                key="document_title",
                placeholder="Enter the heading from the page",
            )
            items = st.text_area(
                "Content — put each list item on a separate line",
                key="document_items",
                height=300,
                help="Edit the recognized words, remove OCR errors, and keep one list item per line.",
            )
            with st.container(border=True):
                st.caption("Document preview")
                if title.strip():
                    st.subheader(title.strip())
                preview_items = [line.strip() for line in items.splitlines() if line.strip()]
                if preview_items:
                    st.markdown("\n".join(f"- {line}" for line in preview_items))
                else:
                    st.caption("Add list items above to preview the document.")
            c1, c2, c3 = st.columns(3)
            c1.download_button(
                "⬇ TXT",
                to_professional_txt(title, items),
                "converted.txt",
                "text/plain",
            )
            c2.download_button("⬇ Word (.docx)", to_professional_docx(title, items), "converted.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            c3.download_button(
                "⬇ PDF",
                to_professional_pdf(title, items),
                "converted.pdf",
                "application/pdf",
            )

with st.expander("⚠️ Responsible use"):
    st.markdown("""
- AI can misread messy handwriting, names, numbers or dates - **always review the text before use**.
- Do not upload documents with private data (IDs, medical, exam papers) unless you have permission.
- Processing runs locally on this computer; nothing is sent to external servers.
""")
