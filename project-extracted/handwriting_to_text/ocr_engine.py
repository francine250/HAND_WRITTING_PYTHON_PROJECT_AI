"""Steps 2, 4, 5 - AI approach: pretrained TrOCR (Vision Transformer encoder + text Transformer decoder)."""
import json, logging, os, time
import cv2
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from preprocessing import preprocess, segment_lines

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/activity.log", level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


def load_config(path="config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class HandwritingOCR:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.get("use_saved_model", False):
            src = cfg["saved_model_dir"]
            if not os.path.isdir(src):
                raise FileNotFoundError(
                    f"Saved model directory does not exist: {src}. "
                    "Disable 'Use a locally saved model instead' or create the directory."
                )
        else:
            src = cfg["model_name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(src)
        self.model = VisionEncoderDecoderModel.from_pretrained(src).to(self.device).eval()
        logging.info("Model loaded from %s on %s", src, self.device)

    @torch.no_grad()
    def recognize_lines(self, line_imgs):
        """line_imgs: list of grayscale numpy arrays -> list of strings."""
        out, bs = [], self.cfg["batch_size"]
        for i in range(0, len(line_imgs), bs):
            batch = [Image.fromarray(l).convert("RGB") for l in line_imgs[i:i + bs]]
            px = self.processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
            ids = self.model.generate(px, num_beams=self.cfg["num_beams"],
                                      max_new_tokens=self.cfg["max_new_tokens"])
            out += [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return out

    def transcribe_page(self, img_bgr, single_line=False):
        """Full pipeline for one page. Returns dict(lines, text, seconds, line_images)."""
        t0 = time.time()
        gray, ink = preprocess(img_bgr, self.cfg)
        line_imgs = [gray] if single_line else segment_lines(gray, ink, self.cfg)
        if not line_imgs:                       # guardrail: nothing detected -> try whole page
            line_imgs = [gray]
            logging.warning("No text lines detected; falling back to whole image")
        lines = self.recognize_lines(line_imgs)
        dt = time.time() - t0
        logging.info("Page transcribed: %d lines in %.2fs", len(lines), dt)
        return {"lines": lines, "text": "\n".join(lines), "seconds": dt, "line_images": line_imgs}

    def save(self, directory=None):
        """Save model + processor so the application can be reproduced offline."""
        directory = directory or self.cfg["saved_model_dir"]
        self.model.save_pretrained(directory)
        self.processor.save_pretrained(directory)
        logging.info("Model saved to %s", directory)
