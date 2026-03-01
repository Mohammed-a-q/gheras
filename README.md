# غراس - مشروع مدرسي

مشروع بسيط لعرض أمام لجنة مدرسية: رفع صورة لمكان، تحليل بسيط باستخدام موديل Vision جاهز، وإظهار قرار + سبب + اقتراح شجرة.

## تشغيل محلي

1. فتح PowerShell داخل المجلد المشروع:
   ```powershell
   cd c:\Users\moham\OneDrive\Documents\mobile-apps\afforestation\tree-project
   ```

2. إنشاء وتفعيل بيئة افتراضية (Windows PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate
```

3. تثبيت المتطلبات:

```powershell
pip install -r requirements.txt
```

4. (اختياري) تنزيل الموديل مسبقًا:

```powershell
python download_model.py
```

5. تشغيل السيرفر:

```powershell
uvicorn main:app --reload
```

6. افتح المتصفح:

http://127.0.0.1:8000

ملاحظة: في التشغيل الأول، سيقوم Transformers بتحميل الموديل (~200-500MB) ثم يعمل دون إنترنت لاحقًا.
