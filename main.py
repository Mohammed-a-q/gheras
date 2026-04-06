import os
import uuid
import logging
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# basic logging
logging.basicConfig(level=logging.INFO)

# Ensure upload directory exists
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.info(f"App starting, BASE_DIR: {BASE_DIR}")
logging.info(f"Templates dir: {os.path.join(BASE_DIR, 'templates')}")
logging.info(f"Static dir: {os.path.join(BASE_DIR, 'static')}")

# Global model placeholder - loaded lazily on first use
model = None

def get_lanczos_filter():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def load_model():
    import torch
    from transformers import pipeline

    # Limit CPU thread usage to reduce pressure under concurrent requests
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        logging.info("PyTorch thread count set to 1 for CPU stability.")
    except Exception as e:
        logging.warning(f"Unable to limit PyTorch threads: {e}")

    # First valid working model for CPU-only environments
    try:
        logging.info("Attempting to load ResNet-18 model for inference...")
        return pipeline(
            "image-classification",
            model="microsoft/resnet-18",
            device=-1,
        )
    except Exception as e:
        logging.warning(f"ResNet-18 load failed: {e}, falling back to ViT-Base")
        try:
            return pipeline(
                "image-classification",
                model="google/vit-base-patch16-224",
                device=-1,
            )
        except Exception as e2:
            logging.warning(f"ViT-Base load failed: {e2}, falling back to ResNet-50")
            try:
                return pipeline(
                    "image-classification",
                    model="microsoft/resnet-50",
                    device=-1,
                )
            except Exception as e3:
                logging.error(
                    f"All model loads failed: ResNet-18: {e}, ViT-Base: {e2}, ResNet-50: {e3}"
                )
                raise RuntimeError("Unable to load any image classification model")


def get_model():
    global model
    if model is None:
        logging.info("Loading model for the first time...")
        model = load_model()
        logging.info("Model loaded successfully.")
    return model


def render_result_html(image_url, decision, reason, suggested_trees, badge_class, city_display):
    suggestions_html = ""
    if suggested_trees:
        suggestions_html = """
            <ul class="suggestions-list">
              {items}
            </ul>
            <p class="muted" style="font-size:0.85rem;margin-top:4px;">اقتُرحت الأشجار بناءً على المدينة والصورة المرفوعة.</p>
""".format(items="\n".join(f"<li>{t}</li>" for t in suggested_trees))
    else:
        suggestions_html = '<p class="muted">لا توجد اقتراحات لأن الموقع غير مناسب.</p>'

    image_section = f'<div class="image-wrap"><img src="{image_url}" alt="الصورة المرفوعة" /></div>' if image_url else ""

    html_template = """<!doctype html>
<html lang="ar">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>النتيجة - غراس</title>
  <link rel="icon" href="/static/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header class="site-header small">
    <div class="container header-inner">
      <div class="brand">
        <h1>غراس</h1>
      </div>
    </div>
  </header>

  <main class="container">
    <section class="card result-card">
      <h2>نتيجة التحليل</h2>

      {image_section}

      <div class="result-row">
        <div class="badge decision-badge {badge_class}">{decision}</div>
        <div class="result-info">
          <div><strong>المدينة:</strong> {city_display}</div>
          <div><strong>السبب:</strong> {reason}</div>
          <div><strong>اقتراح أشجار:</strong>
            {suggestions_html}
          </div>
        </div>
      </div>
      

      <div class="actions">
        <a class="btn secondary" href="/">🔁 تحليل صورة أخرى</a>
      </div>

      <p class="note">تنبيه: النتائج تقديرية وتعتمد على تحليل آلي، ولا تغني عن استشارة مختص زراعي.</p>
    </section>
  </main>
</body>
</html>"""

    html = html_template.format(
        image_section=image_section,
        badge_class=badge_class,
        decision=decision,
        city_display=city_display,
        reason=reason,
        suggestions_html=suggestions_html
    )

    return HTMLResponse(content=html, status_code=200)


app = FastAPI(title="غراس — مشروع مدرسي")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logging.exception(f"Unhandled exception while processing request: {type(e).__name__}: {str(e)}")
        raise


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
async def index(request: Request):
    try:
        template_path = os.path.join(BASE_DIR, "templates", "index.html")
        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logging.error(f"Error serving app index: {type(e).__name__}: {str(e)}", exc_info=True)
        return HTMLResponse(content="<h1>خطأ في الخادم</h1>", status_code=500)


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
        # Reduce image size to model input size for memory efficiency (224x224 for most models)
        MAX_SIZE = (224, 224)
        image.thumbnail(MAX_SIZE, get_lanczos_filter())
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
            logging.info(f"Color analysis: green_pct={green_pct:.3f}, gray_pct={gray_pct:.3f}")
        except Exception:
            # if numpy or conversion fails, fall back to neutral values
            green_pct = 0.0
            gray_pct = 0.0
    except Exception as e:
        logging.exception("Failed to open/process uploaded image")
        return render_result_html(
            image_url=None,
            decision="خطأ في فتح الصورة",
            reason="الملف المرفوع ليس صورة صالحة.",
            suggested_trees=[],
            badge_class="danger",
            city_display=city,
        )

    image_url = f"/static/uploads/{filename}"

    # Decide primarily by color analysis: green, asphalt/gray, sand.
    decision = None
    suggested_trees = []

    # sand-like pixels: light brown/beige with moderate saturation and high brightness
    try:
        sand_mask = (
            (h >= 5) & (h <= 35) &
            (s >= 20) & (s <= 120) &
            (v >= 120)
        )
        sand_pct = float(sand_mask.mean())
    except Exception:
        sand_pct = 0.0

    logging.info(f"Color analysis: green_pct={green_pct:.3f}, gray_pct={gray_pct:.3f}, sand_pct={sand_pct:.3f}")

    if green_pct >= 0.08:
        decision = "✅ مناسب للتشجير"
        reason = "المنطقة خضراء ظاهريًا؛ البيئة تبدو مناسبة للتشجير."
        suggested_trees = ["نيم"]
        badge_class = "success"
    elif gray_pct >= 0.14:
        decision = "❌ غير مناسب"
        reason = "الصورة تحتوي على مساحات رمادية/داكنة متجانسة مثل أسفلت أو خرسانة، مما يجعل الزراعة غير مناسبة."
        suggested_trees = []
        badge_class = "danger"
    elif sand_pct >= 0.08:
        decision = "⚠️ مناسب بشروط"
        reason = "يظهر في الصورة سطح رملي أو غير مستقر؛ يفضل أشجار مقاومة للجفاف وإجراءات تحضيرية للتربة."
        suggested_trees = ["طلح"]
        badge_class = "warning"
    else:
        # If color cues are weak, use the model as a fallback for a better guess.
        model_fn = get_model()
        try:
            logging.info("Starting fallback model inference...")
            results = model_fn(image, top_k=1)
            logging.info("Model inference completed successfully.")
            image.close()
            import gc
            gc.collect()
        except Exception as e:
            logging.exception("Fallback model inference failed")
            decision = "⚠️ مناسب بشروط"
            reason = "المنطقة غير واضحة في الصورة؛ يوصى بفحص ميداني وتوفير صور إضافية."
            suggested_trees = ["نيم", "شجرة تتحمّل الجفاف"]
            badge_class = "warning"
            # continue to city mapping
            results = []

        if decision is None:
            labels = [r.get("label", "").lower() for r in results]
            joined = " ".join(labels)
            logging.info(f"Model fallback labels: {labels}, joined: {joined}")

            if any(k in joined for k in ["tree", "forest", "grass", "lawn", "park"]):
                decision = "✅ مناسب للتشجير"
                reason = "التحليل الألي يشير إلى عناصر نباتية واضحة في الصورة."
                suggested_trees = ["نيم"]
                badge_class = "success"
            elif any(k in joined for k in ["asphalt", "road", "pavement", "concrete", "building"]):
                decision = "❌ غير مناسب"
                reason = "التحليل الألي يشير إلى معالم صلبة مثل طرق أو مبانٍ."
                suggested_trees = []
                badge_class = "danger"
            elif any(k in joined for k in ["sand", "desert", "dune", "soil", "dirt"]):
                decision = "⚠️ مناسب بشروط"
                reason = "التحليل الألي يشير إلى سطح رملي؛ يفضل اختيار أنواع تتحمل الجفاف."
                suggested_trees = ["طلح"]
                badge_class = "warning"
            else:
                decision = "⚠️ مناسب بشروط"
                reason = "المنطقة غير واضحة بشكل كافٍ في الصورة، نوصي بالفحص الميداني أو مواصلة جمع صور أخرى."
                suggested_trees = ["نيم", "شجرة تتحمّل الجفاف"]
                badge_class = "warning"

    # close the image after classification to free resources
    try:
        image.close()
    except Exception:
        pass

    # map city code to Arabic display name and list of suggestions
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

    if "غير مناسب" not in decision:
        suggested_trees = city_tree_map.get(city, suggested_trees)
    else:
        suggested_trees = []
    city_display = city_display_map.get(city, city)

    if decision.startswith("❌") or "غير مناسب" in decision:
        badge_class = "danger"
    elif decision.startswith("⚠️") or "مناسب بش" in decision:
        badge_class = "warning"
    else:
        badge_class = "success"

    return render_result_html(
        image_url=image_url,
        decision=decision,
        reason=reason,
        suggested_trees=suggested_trees,
        badge_class=badge_class,
        city_display=city_display,
    )


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
