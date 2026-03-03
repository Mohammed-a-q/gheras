import os
import uuid
import logging
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

# basic logging
logging.basicConfig(level=logging.INFO)

# Ensure upload directory exists
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global classifier placeholder - loaded lazily on first use
classifier = None

def get_classifier():
    """Load classifier lazily on first use.

    The primary model is ViT-base. If loading that model fails (e.g. due to
    out‑of‑memory on Render or similar), fall back to a much smaller ResNet
    model so that inference can still proceed.  We only allocate the pipeline
    once and cache it globally.
    """
    global classifier
    if classifier is None:
        from transformers import pipeline
        try:
            classifier = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224",
                device=-1  # CPU only
            )
        except Exception:
            logging.exception("Primary model load failed, falling back to ResNet-50")
            # smaller model that usually fits in limited memory
            classifier = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device=-1
            )
    return classifier

app = FastAPI(title="غراس — مشروع مدرسي")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/health")
async def health():
    """Simple health check endpoint (non-blocking, for Railway monitoring)"""
    return {"status": "ok"}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...), city: str = Form(...)):
    # save uploaded file
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    # Open image and run classifier
    try:
        image = Image.open(file_path)
        # reduce image size to save memory during analysis (Render-friendly)
        MAX_SIZE = (512, 512)
        image.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
        image = image.convert("RGB")
        # --- Color heuristic (fallback) ---
        # compute simple green / gray (asphalt) percentages to help
        # override model when it's uncertain or when surface is clearly asphalt/grass
        try:
            hsv = np.array(image.convert("HSV"))
            h = hsv[:, :, 0].astype(int)
            s = hsv[:, :, 1].astype(int)
            v = hsv[:, :, 2].astype(int)
            # green hue roughly between ~35..100 on 0-255 HSV scale
            green_mask = (h >= 35) & (h <= 100) & (s >= 40) & (v >= 40)
            green_pct = float(green_mask.mean())

            arr = np.array(image)
            r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
            rgb_diff = (np.abs(r - g) + np.abs(r - b) + np.abs(g - b))
            # gray/asphalt: low color variance and relatively low brightness
            gray_mask = (rgb_diff <= 30) & (v <= 120)
            gray_pct = float(gray_mask.mean())
        except Exception:
            # if numpy or conversion fails, fall back to neutral values
            green_pct = 0.0
            gray_pct = 0.0
    except Exception as e:
        logging.exception("Failed to open/process uploaded image")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "image_url": None,
            "decision": "خطأ في فتح الصورة",
            "reason": "الملف المرفوع ليس صورة صالحة.",
            "suggested_trees": [],
            "badge_class": "danger",
            "city": city
        })

    # image color analysis removed (feature disabled)

    image_url = f"/static/uploads/{filename}"

    # Load and run classifier (lazy loading on first request)
    classifier_fn = get_classifier()
    try:
        # use smaller top_k to reduce memory & bandwidth; 3 still gives good clues
        results = classifier_fn(image, top_k=3)
    except Exception as e:
        logging.exception("Model inference failed")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "image_url": image_url,
            "decision": "خطأ في التحليل",
            "reason": "حدث خطأ أثناء تحليل الصورة على الخادم. الرجاء المحاولة لاحقًا.",
            "suggested_trees": [],
            "badge_class": "danger",
            "city": city
        })
    labels = [r.get("label", "").lower() for r in results]
    joined = " ".join(labels)

    # Color-based overrides: give priority to clear green/asphalt images
    color_override = False
    # if image is clearly dominated by gray/dark uniform pixels -> likely asphalt/road
    if gray_pct > 0.12:
        decision = "❌ غير مناسب"
        reason = "تحتوي الصورة على مساحات صلبة (مثل أسفلت أو أرضيات خرسانية) مما يجعل الزراعة غير مناسبة."
        suggested_trees = []
        color_override = True
    # if image shows a clear green cover -> prefer 'مناسب' regardless of weak model signals
    elif green_pct > 0.08:
        decision = "✅ مناسب للتشجير"
        reason = "المنطقة خضراء ظاهريًا؛ البيئة مناسبة للتشجير." 
        suggested_trees = ["نيم"]
        color_override = True

    # simple keyword mapping (Arabic UI; logic based on English labels from model)
    suitable_kw = ["grass", "tree", "trees", "garden", "lawn", "park", "plant", "foliage"]
    conditional_kw = ["desert", "sand", "dune"]
    unsuitable_kw = ["road", "asphalt", "building", "concrete", "pavement", "car", "vehicle"]

    decision = "غير مُحدد"
    reason = "لم يتم التعرف على معالم كافية في الصورة لاتخاذ قرار قاطع."
    suggested_trees = []

    # prioritize: unsuitable > suitable > conditional
    if any(k in joined for k in unsuitable_kw):
        decision = "❌ غير مناسب"
        reason = "تحتوي الصورة على معالم صلبة (أسفلت/طرق/مبانٍ) مما يجعل الزراعة غير مناسبة." 
        # unsuitable sites get no tree suggestions
        suggested_trees = []
    elif any(k in joined for k in suitable_kw):
        decision = "✅ مناسب للتشجير"
        reason = "تظهر عناصر نباتية مثل عشب/أشجار في الصورة؛ المكان يبدو مناسباً للتشجير." 
        suggested_trees = ["نيم"]
    elif any(k in joined for k in conditional_kw):
        decision = "⚠️ مناسب بشروط"
        reason = "المنطقة رملية/صحراوية؛ يوصى بأشجار مقاومة للجفاف وإجراءات تحضيرية للتربة." 
        suggested_trees = ["طلح"]
    else:
        # fallback: not enough clear features identified
        decision = "⚠️ مناسب بشروط"
        reason = "لم تتضح معالم كافية في الصورة؛ ينصح بفحص ميداني وإدخال مزيد من الصور للحصول على تقييم أدق."
        suggested_trees = ["نيم", "شجرة تتحمّل الجفاف"]

    # (color-based adjustments removed)

    image_url = f"/static/uploads/{filename}"

    # map city code to Arabic display name and list of suggestions
    # suggestion criterion:
    # 1. Base on classification/colour analysis decision above
    # 2. For suitable/conditional locations, we pick a few tree names that
    #    generally thrive in the selected city/climate.
    # 3. The city_tree_map is a simple lookup: we recommend three local species
    #    with preference for heat or drought tolerance.
    # 4. No suggestions are returned when decision indicates "غير مناسب".
    city_display_map = {
        "Riyadh": "الرياض",
        "Jeddah": "جدة",
        "Dammam": "الدمام",
        "Abha": "أبها",
        "Tabuk": "تبوك"
    }

    city_tree_map = {
        "Riyadh": ["نيم", "طلح", "سدر"],
        "Jeddah": ["سدر", "نخيل", "طلح"],
        "Dammam": ["نخيل", "طلح", "نيم"],
        "Abha": ["سدر", "طلح", "شجرة ظل محلية"],
        "Tabuk": ["طلح", "نيم", "شجرة مقاومة للجفاف"]
    }

    # apply city-specific list only if location still considered suitable/conditional
    if "غير مناسب" not in decision:
        suggested_trees = city_tree_map.get(city, suggested_trees)
    else:
        suggested_trees = []
    city_display = city_display_map.get(city, city)

    # decide badge class for UI (success / warning / danger)
    badge_class = "success"
    if decision.startswith("❌") or "غير مناسب" in decision:
        badge_class = "danger"
    elif decision.startswith("⚠️") or "مناسب بش" in decision:
        badge_class = "warning"

    return templates.TemplateResponse("result.html", {
        "request": request,
        "image_url": image_url,
        "decision": decision,
        "reason": reason,
        "suggested_trees": suggested_trees,
        "badge_class": badge_class,
        "city": city_display
    })
