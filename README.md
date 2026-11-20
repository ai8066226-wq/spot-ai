# Spot AI — تدريب Python على GitHub Actions

واجهة Kotlin مؤجلة. هذا المستودع يدرب النموذج ويختبر المؤشرات وينتج ملفات للتنزيل، دون Termux أو خادم دائم أو مفاتيح Bybit.

## من الهاتف
1. أنشئ مستودع GitHub جديدًا أو اختر مستودعًا مخصصًا لهذا المشروع.
2. فك ضغط الحزمة وارفع محتويات مجلد spot_ai_github إلى جذر المستودع، وليس ملف ZIP نفسه. يجب وجود engine.py وrequirements.txt في الجذر، وملف research.yml داخل .github/workflows/.
3. إذا لم يعرض مدير ملفات الهاتف المجلد المخفي .github: افتح Add file ثم Create new file على GitHub واكتب الاسم الكامل .github/workflows/research.yml وانسخ محتواه من WORKFLOW_COPY.txt. هذا الملف نسخة مرئية للتسهيل فقط.
4. احفظ الملفات في الفرع الافتراضي، ثم افتح Actions واسمح بتشغيل Workflows إن ظهر الطلب.
5. اختر Spot AI Research ثم Run workflow. اترك source=synthetic في أول تجربة، ثم شغل المهمة.
6. بعد نجاحها، افتح صفحة التشغيل ثم Artifacts ونزل spot-ai-synthetic-رقم_التشغيل.
7. للتدريب على بيانات سوق فعلية شغل مهمة جديدة واختر source=bybit وsymbol=BTCUSDT والفاصل 15 وdays=180.

لا تحتاج أسرارًا ولا API key لجلب بيانات السوق العامة. تعديل الرسوم والانزلاق مسؤولية إعداد التجربة؛ القيم المبدئية ليست بيانًا برسوم حسابك.

## ما تحمله بعد التشغيل؟
- report.json: ترتيب المؤشرات، نوافذ التحقق، أسباب الاستبعاد، نتائج النموذج، ومقارنة الشراء والاحتفاظ.
- model.pkl: النموذج البحثي؛ لا يحمّل مباشرة داخل Kotlin. تحويله وربطه بالتطبيق مرحلة لاحقة.
- candles.csv: البيانات المستخدمة.
- run.json: مصدر البيانات والإعدادات وإصدارات المكتبات وبصمة البيانات ومرجع الكود.
- SUMMARY.md: ملخص النتائج. بيانات synthetic لا تثبت الربحية، وعلامة model_approved تعطل معها دائمًا.
- error.txt: تفاصيل الفشل إن حصل.

النتائج تحفظ لمدة 14 يومًا وفق إعداد المهمة، ضمن حدود حسابك. نزلها قبل انتهاء المدة. لا توجد جدولة تلقائية أو تداول حقيقي.

قد تمنع Bybit الوصول من موقع خادم GitHub. عندها تفشل المهمة بوضوح، ولا تستبدل البيانات ببيانات اصطناعية. نراجع الخطأ ونختار مصدر بيانات متاحًا ومصرحًا أو ملف تاريخي موثوق في مرحلة لاحقة؛ لا نغيّر المنطقة للتحايل على القيود.

راجع ENGINE_AR.md لشرح منهجية الاختبار وحدود الإصدار. لا تكرر ضبط الاستراتيجية اعتمادًا على فترة الاختبار النهائي.

## تحقق محلي اختياري
```bash
python -m pip install --only-binary=:all: -r requirements.txt
python -m unittest discover -s tests -v
python run_research.py
```

المراجع:
- https://bybit-exchange.github.io/docs/v5/market/kline
- https://docs.github.com/en/actions/how-tos/manage-workflow-runs
- https://docs.github.com/en/actions/tutorials/store-and-share-data

هذه الحزمة مجهزة للتشغيل على GitHub؛ التشغيل المحلي الاصطناعي لا يعني أنها نُفذت بالفعل داخل حساب GitHub.
