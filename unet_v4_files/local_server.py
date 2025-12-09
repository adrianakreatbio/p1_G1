# local_server.py
import io
from pathlib import Path
from typing import List
import torch
import numpy as np
import imageio.v3 as iio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont
from u_net import GelUNet, concentrations_relative_to_lane1, preprocess_gel, predict_mask

app = FastAPI()


# Automatically choose GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "trained_weights.pth"

_model: GelUNet | None = None

def get_model() -> GelUNet:
    """Load the trained GelUNet model once into memory."""
    global _model
    if _model is None:
        model = GelUNet(in_channels=1, out_channels=1)
        state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model

def infer_all(image_path: str, standard_conc: float):
    model = get_model()
    img = iio.imread(image_path)
    res = concentrations_relative_to_lane1(img=img, model=model, n_lanes=None, lane1_conc=standard_conc)
    return res["lanes"], res["absolute"], res["bands_abs"]


def run_inference(image_path: str, standard_conc: float) -> List[float]:
    """Return per-lane lists of per-band concentrations."""
    _, _, bands_abs = infer_all(image_path, standard_conc)
    # ensure JSON-friendly floats
    return [[float(b) for b in lane] for lane in bands_abs]


def build_pdf(lane_concs, standard_conc: float, image_filename: str) -> io.BytesIO:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # resolve paths
    ROOT = Path(__file__).resolve().parent
    reports = ROOT / "Reports"
    time_part = Path(image_filename).stem.replace("PhotoCapture_", "").replace("Gel_", "")
    run_dir = reports / f"Data_{time_part}"

    orig = (run_dir / f"Gel_{time_part}.png")
    annotated = (run_dir / f"Annotated_{time_part}.png")
    mask = (run_dir / f"Mask_{time_part}.png")

    # ========== PAGE 1+: CONCENTRATION TABLE (EXTENDS IF NEEDED) ==========
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "G1 DNA Quantification Report")
    
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 80, f"Standard concentration (Lane 1): {standard_conc:.1f} ng/µL")

    # Table header
    y = height - 120
    c.setFont("Helvetica-Bold", 11)
    col1_x, col2_x, col3_x = 50, 200, 350
    col_spacing = 20
    
    c.drawString(col1_x, y, "Lane")
    c.drawString(col2_x, y, "Total (ng/µL)")
    c.drawString(col3_x, y, "Individual Band (ng/µL)")
    
    # Horizontal line under header
    y -= 5
    c.line(50, y, 550, y)
    y -= col_spacing

    # Table rows
    c.setFont("Helvetica", 10)
    bottom_margin = 50  # space to leave at bottom
    
    for lane_idx, bands in enumerate(lane_concs, start=1):
        # Check if we need a new page
        if y < bottom_margin:
            # Horizontal line at bottom of current page
            c.line(50, y + col_spacing, 550, y + col_spacing)
            c.showPage()
            # Start new page with table header again
            y = height - 50
            c.setFont("Helvetica-Bold", 11)
            c.drawString(col1_x, y, "Lane")
            c.drawString(col2_x, y, "Total (ng/µL)")
            c.drawString(col3_x, y, "Individual Band (ng/µL)")
            y -= 5
            c.line(50, y, 550, y)
            y -= col_spacing
            c.setFont("Helvetica", 10)
        
        total = sum(bands) if bands else 0
        bands_str = ", ".join([f"{b:.1f}" for b in bands]) if bands else "-"
        
        c.drawString(col1_x, y, str(lane_idx))
        c.drawString(col2_x, y, f"{total:.1f}")
        c.drawString(col3_x, y, bands_str)
        y -= col_spacing
    
    # Horizontal line under table
    c.line(50, y, 550, y)
    c.showPage()

    # ========== NEXT PAGE: ORIGINAL GEL IMAGE (FULL PAGE) ==========
    if orig.exists():
        try:
            from PIL import Image as PILImage
            img = PILImage.open(orig)
            img_w, img_h = img.size
            
            # Calculate scaling to fit page (with margins)
            margin = 40
            max_w = width - 2 * margin
            max_h = height - 2 * margin
            scale = min(max_w / img_w, max_h / img_h)
            
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            # Position
            x = (width - new_w) / 2
            y_center = height - margin - 30 - new_h
            
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, height - margin, "Original gel image")
            
            c.drawImage(str(orig), x, y_center, width=new_w, height=new_h, preserveAspectRatio=True, mask='auto')
            c.showPage()
        except Exception as e:
            print(f"Error adding original image: {e}")
            c.showPage()

    # ========== NEXT PAGE: ANNOTATED IMAGE (TOP) AND MASK IMAGE (BOTTOM) ==========
    margin = 40
    max_img_w = width - 2 * margin
    half_h = (height - 150) / 2

    # Annotated image (top)
    if annotated.exists():
        try:
            from PIL import Image as PILImage
            img = PILImage.open(annotated)
            img_w, img_h = img.size
            scale = min(max_img_w / img_w, half_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            x = (width - new_w) / 2
            
            # Position from top
            y_anno = height - margin - 30 - new_h  # Start from top with title space
            
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, height - margin, "Annotated gel image")
            c.drawImage(str(annotated), x, y_anno, width=new_w, height=new_h, preserveAspectRatio=True, mask='auto')
        except Exception as e:
            print(f"Error adding annotated image: {e}")
        c.showPage()

    # Mask image new page
    if mask.exists():
        try:
            from PIL import Image as PILImage
            img = PILImage.open(mask)
            img_w, img_h = img.size
            scale = min(max_img_w / img_w, half_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            x = (width - new_w) / 2
            
            # Position directly below annotated image with spacing
            y_mask = height - margin - 30 - new_h  # Start from top with title space
            
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, height - margin, "Mask image")
            c.drawImage(str(mask), x, y_mask, width=new_w, height=new_h, preserveAspectRatio=True, mask='auto')
        except Exception as e:
            print(f"Error adding mask image: {e}")
        c.showPage()

    c.save()
    buffer.seek(0)
    return buffer


def build_annotated_image(image_path: str, standard_conc: float) -> io.BytesIO:
    """Run the model, then draw simple lane labels L1, L2, ... on top of the gel image."""

    img = iio.imread(image_path)
    lanes, _, _ = infer_all(image_path, standard_conc)

    # Ensure RGB for drawing
    if img.ndim == 2:
        img_rgb = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img_rgb = img[..., :3]
    else:
        img_rgb = img

    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    h = pil_img.height
    font_size = max(h // 18, 24)

    try:
        # works on most systems (Pillow ships DejaVu fonts)
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            # fallback: default bitmap font (size fixed and small)
            font = ImageFont.load_default()

    # vertical position for labels (8% from top)
    y_top = int(0.08 * h)

    for i, (x0, x1) in enumerate(lanes, start=1):
        label = f"L{i}"
        xc = int(0.5 * (x0 + x1))  # center of lane

        # get text size; font.getbbox works for TrueType
        try:
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            # older Pillow
            text_w, text_h = draw.textsize(label, font=font)

        draw.text((xc - text_w // 2, y_top - text_h // 2), label, font=font, fill=(255, 255, 255),)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

"""
@app.post("/analyze")
async def analyze(standard_conc: float = Form(...), file: UploadFile = File(...)):
    ROOT = Path(__file__).resolve().parent
    reports_dir = ROOT / "Reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    img_path = reports_dir / file.filename
    with open(img_path, "wb") as f:
        f.write(await file.read())

    base = Path(file.filename).stem
    time_part = base.replace("PhotoCapture_", "").replace("Gel_", "")
    run_dir = reports_dir / f"Data_{time_part}"  # may or may not exist

    lane_concs = run_inference(str(img_path), standard_conc)  # [[band1, ...], ...]
    pdf_buffer = build_pdf(lane_concs, standard_conc, file.filename)

    pdf_name = f"Report_{time_part}.pdf"
    return StreamingResponse(pdf_buffer, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename={pdf_name}"},)
"""

@app.post("/analyze")
async def analyze(standard_conc: float = Form(...), file: UploadFile = File(...)):
    ROOT = Path(__file__).resolve().parent
    reports_dir = ROOT / "Reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    img_path = reports_dir / file.filename
    with open(img_path, "wb") as f:
        f.write(await file.read())

    base = Path(file.filename).stem
    time_part = base.replace("PhotoCapture_", "").replace("Gel_", "")
    run_dir = reports_dir / f"Data_{time_part}"
    run_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # ========== GENERATE ANNOTATED IMAGE ==========
    annotated_buffer = build_annotated_image(str(img_path), standard_conc)
    annotated_path = run_dir / f"Annotated_{time_part}.png"
    with open(annotated_path, "wb") as f:
        f.write(annotated_buffer.getvalue())
    
    # ========== GENERATE MASK IMAGE ==========
    model = get_model()
    img = iio.imread(img_path)
    res = concentrations_relative_to_lane1(img=img, model=model, n_lanes=None, lane1_conc=standard_conc)
    mask_np = res["mask_vis"]
    mask_img = (mask_np * 255).astype(np.uint8)
    mask_path = run_dir / f"Mask_{time_part}.png"
    Image.fromarray(mask_img).save(mask_path, format="PNG")

    # ========== COPY ORIGINAL GEL IMAGE ==========
    gel_path = run_dir / f"Gel_{time_part}.png"
    if not gel_path.exists():
        import shutil
        shutil.copy(str(img_path), str(gel_path))

    # ========== NOW BUILD PDF (all images exist) ==========
    lane_concs = run_inference(str(img_path), standard_conc)
    pdf_buffer = build_pdf(lane_concs, standard_conc, file.filename)

    pdf_name = f"Report_{time_part}.pdf"
    return StreamingResponse(pdf_buffer, media_type="application/pdf", 
                           headers={"Content-Disposition": f"attachment; filename={pdf_name}"})

@app.post("/analyze_json")
async def analyze_json(standard_conc: float = Form(...), file: UploadFile = File(...)):
    ROOT = Path(__file__).resolve().parent
    reports_dir = ROOT / "Reports"; reports_dir.mkdir(parents=True, exist_ok=True)
    img_path = reports_dir / file.filename
    with open(img_path, "wb") as f: f.write(await file.read())
    lane_concs = run_inference(str(img_path), standard_conc)  # [[band1, band2, ...], ...]
    return JSONResponse({"lane_concs": lane_concs})



@app.post("/annotate")
async def annotate(standard_conc: float = Form(...), file: UploadFile = File(...)):
    tmp_path = Path("temp_files/temp_gel_image_for_annotate.png")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        # build annotated bytes
        img_buffer = build_annotated_image(str(tmp_path), standard_conc)

        # ALSO SAVE TO Reports/Data_<timePart>/Annotated_<timePart>.png
        ROOT = Path(__file__).resolve().parent
        reports = ROOT / "Reports"
        reports.mkdir(parents=True, exist_ok=True)
        base = Path(file.filename).stem
        time_part = base.replace("PhotoCapture_", "").replace("Gel_", "")
        run_dir = reports / f"Data_{time_part}"
        run_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = run_dir / f"Annotated_{time_part}.png"
        with open(annotated_path, "wb") as out:
            out.write(img_buffer.getvalue())

        return StreamingResponse(
            io.BytesIO(img_buffer.getvalue()),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=gel_annotated.png"},
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()



@app.post("/mask")
async def mask(standard_conc: float = Form(...), file: UploadFile = File(...)):
    tmp_path = Path("temp_files/temp_gel_image_for_mask.png")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        model = get_model()
        img = iio.imread(tmp_path)
        res = concentrations_relative_to_lane1(img=img, model=model, n_lanes=None, lane1_conc=standard_conc)
        mask_np = res["mask_vis"]
        mask_img = (mask_np * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(mask_img).save(buf, format="PNG")

        # SAVE copy to Reports/Data_<timePart>/Mask_<timePart>.png
        ROOT = Path(__file__).resolve().parent
        reports = ROOT / "Reports"
        reports.mkdir(parents=True, exist_ok=True)
        base = Path(file.filename).stem
        time_part = base.replace("PhotoCapture_", "").replace("Mask_", "")
        run_dir = reports / f"Data_{time_part}"
        run_dir.mkdir(parents=True, exist_ok=True)
        mask_path = run_dir / f"Mask_{time_part}.png"
        with open(mask_path, "wb") as out:
            out.write(buf.getvalue())

        buf.seek(0)
        return StreamingResponse(
            buf, media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=gel_mask.png"},
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.get("/get_annotated/{time_part}")
async def get_annotated(time_part: str):
    """Return the annotated image for a given time_part."""
    ROOT = Path(__file__).resolve().parent
    reports_dir = ROOT / "Reports"
    run_dir = reports_dir / f"Data_{time_part}"
    annotated_path = run_dir / f"Annotated_{time_part}.png"
    
    if not annotated_path.exists():
        return JSONResponse({"error": "Annotated image not found"}, status_code=404)
    
    return FileResponse(annotated_path, media_type="image/png")


@app.get("/get_mask/{time_part}")
async def get_mask(time_part: str):
    """Return the mask image for a given time_part."""
    ROOT = Path(__file__).resolve().parent
    reports_dir = ROOT / "Reports"
    run_dir = reports_dir / f"Data_{time_part}"
    mask_path = run_dir / f"Mask_{time_part}.png"
    
    if not mask_path.exists():
        return JSONResponse({"error": "Mask image not found"}, status_code=404)
    
    return FileResponse(mask_path, media_type="image/png")




# WORKFLOW
# RECEIVE FASTAPI SIGNAL TO ANALYZE -> run_inference (get_model) -> build pdf

# NOTES
# @: tell fastAPI to run the def analyze if POST request is  sent to /analyze
## to delete temporary files - put in the correct folders




