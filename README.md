# غراس - مشروع مدرسي

مشروع بسيط لعرض أمام لجنة مدرسية: رفع صورة لمكان، تحليل بسيط باستخدام موديل Vision جاهز، وإظهار قرار + سبب + اقتراح شجرة.

## التحسينات الجديدة (Render-Ready)

✅ **Python 3.10**: تحديد إصدار 3.10 للتوافق مع مكتبات المشروع  
✅ **Lazy Loading**: النموذج يُحمّل فقط عند أول طلب تحليل (توفير ذاكرة عند البدء)  
✅ **تقليل الصور**: كل الصور تُعاد تحجيمها إلى 512×512 قبل التحليل  
✅ **CPU-Only**: توافق كامل مع CPU بدون CUDA dependencies  
✅ **Single Worker**: Uvicorn يعمل بـ worker واحد فقط لتقليل استهلاك الذاكرة

> تم إزالة تثبيت `tokenizers` من المتطلبات بسبب حاجته لأداوات `rust`، وهذا حل فشل البناء السابق على Render.

## التشغيل على Render

1. ادفع المشروع إلى GitHub (تم بالفعل)
2. توجّه إلى https://dashboard.render.com
3. انقر **New +** → **Web Service**
4. اربط مع `https://github.com/Mohammed-a-q/gheras`
5. اختر الخطة المجانية (Free)
6. انقر **Deploy**

Render سيقرأ `render.yaml` تلقائيًا ويطبق الإعدادات الصحيحة.

**ملاحظة**: التحميل الأول قد يستغرق 2-3 دقائق (تحميل الموديل من HuggingFace).

## التشغيل محليًا (Windows)

### 1. إعداد البيئة

```powershell
cd c:\Users\moham\OneDrive\Documents\mobile-apps\afforestation\tree-project
python -m venv venv
.\venv\Scripts\Activate
```

### 2. تثبيت المتطلبات

```powershell
pip install -r requirements.txt
```

### 3. (اختياري) تنزيل الموديل مسبقًا

```powershell
python download_model.py
```

هذا يوفر وقتًا قيّمًا عند العرض الأول للجنة المدرسية.

### 4. تشغيل السيرفر

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. افتح المتصفح

http://127.0.0.1:8000

## البنية

```
tree-project/
├── main.py                 # FastAPI app مع lazy loading  
├── templates/              # HTML templates (Jinja2)
│   ├── index.html
│   └── result.html
├── static/                 # CSS, SVG, uploads
│   ├── style.css
│   ├── favicon.svg
│   ├── header-hero.svg
│   └── uploads/           # uploaded images
├── download_model.py       # script لتحميل الموديل مسبقًا
├── requirements.txt        # dependencies (CPU-only)
├── render.yaml            # Render deployment config
└── README.md              # هذا الملف
```

## الملاحظات

- **CPU-Only**: يعمل على Render (free plan بدون GPU)
- **Memory Optimized**: تقليل حجم الصور + lazy loading + single worker
- **Offline-Capable**: بعد أول تحميل للموديل، يعمل بدون إنترنت
- **العربية**: واجهة كاملة باللغة العربية
