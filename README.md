# Spot AI 0.3 — بحث تداول فوري على GitHub Actions

Python لتدريب النموذج واختبار المؤشرات. واجهة Kotlin مؤجلة حتى رفع مشروع التطبيق.
لا تنفيذ صفقات، ولا مفاتيح حساب، ولا ضمان أرباح. اقرأ RESEARCH_AR.md للمنهجية والمصادر والحدود.

## التشغيل من الهاتف
Actions → Spot AI Research → Run workflow.
- synthetic: تجربة برمجية ببيانات اصطناعية فقط، وليس اختبار ربحية.
- bybit: جلب بيانات السوق العامة. آخر محاولة في هذا الحساب توقفت بـ403؛ إعادة التشغيل لا تضمن إزالة الحظر.
- csv: تدريب على ملف شموع ترفعه من مصدر مسموح إلى data/candles.csv. اكتب المسار في data_file والزوج في symbol والفاصل بالدقائق في interval.

اترك الفرع main. إعدادات الرسوم والانزلاق كسور لكل طرف (0.001 = 0.1%)؛ لا نفترض أنها رسوم حسابك الفعلية.
في source=csv نستخدم كل الملف كما هو، ولا تقصه خانة days. يجب أن يطابق الفاصل المحدد، مع 2600 شمعة مغلقة متتابعة على الأقل (يفضل تاريخ أطول). بيانات Bybit تستخدم days فعليًا.
صيغة CSV:
```csv
timestamp,open,high,low,close,volume
```
timestamp هو وقت بداية الشمعة بالميلي ثانية UTC. لا ترفع مفاتيح API أو بيانات شخصية مع الملف.

## المخرجات تحت Artifacts
- report.json: مقارنة 24 إعدادًا، نوافذ التحقق، مقاييس الربح والخسارة والتراجع، وقرار القبول/الرفض.
- SUMMARY.md: ملخص القرار؛ يظهر أيضًا في صفحة التشغيل.
- run.json: المصدر والإعدادات وإصدارات المكتبات وبصمة البيانات ومرجع الكود.
- candles.csv: الشموع المستخدمة.
- model.pkl وtrades.csv وequity.csv وpredictions.csv: تظهر فقط إذا اجتاز مرشح مرحلة اختيار النموذج، حتى لو رُفض في التقييم النهائي. ملف النموذج يحمل live_enabled=false دائمًا؛ لا يعمل مباشرة في Kotlin.
- لا يوجد model.pkl عندما تفشل كل الإعدادات أو عندما تكون التجربة اصطناعية. هذا متوقع ولا يعني فشل البرنامج.
- error.txt عند حصول خطأ.

REJECTED_KEEP_CASH = لا توجد أدلة كافية وفق شروط البحث؛ لا نخفف الشروط تلقائيًا للحصول على ربح ظاهري.
PAPER_CANDIDATE_ONLY = اجتاز شروط البحث الأولية، ويحتاج تجربة ورقية مستقبلية.
SYNTHETIC_TEST_ONLY = تحقق برمجي فقط.
الملفات تحفظ 14 يومًا في مهمة البحث، وفحص Verify Spot AI يحتفظ بتجربته الاصطناعية 7 أيام.

## مكونات الإصدار
research_v3.py: النموذج المعاير، التحقق المتتابع، المحاكي وإدارة المخاطر.
engine.py: جلب البيانات وحساب المؤشرات، مع إبقاء محرك v0.1 للمقارنة التاريخية فقط. تشغيل engine.py research مباشرة يستعمل المنهجية القديمة؛ المسار الجديد هو run_research.py.
ENGINE_AR.md يوثق الإصدار القديم ولا يصف قرارات المحرك الجديد.
WORKFLOW_COPY.txt نسخة مرئية لملف workflow لمن لا يعرض هاتفه مجلد .github المخفي.

## تحقق محلي اختياري (ليس مطلوبًا على الهاتف)
```bash
python -m pip install --only-binary=:all: -r requirements.txt
python -m unittest discover -s tests -v
python run_research.py
```

شغّل التدريب عند الحاجة يدويًا. لا يوجد خادم دائم أو إعادة تدريب مجدولة أو اتصال بحساب التداول.

## تجربة توقع العائد الصافي

expected_return.py يقارن نموذجين بسيطين بمتوسط ثابت، مع تقدير Q10 للخسارة. اقرأ reports/ICNT_EXPECTED_RETURN_AR.md. المهمة Evaluate ICNT Expected Return تحفظ نموذجًا بحثيًا وتقريرًا؛ النتيجة الحالية مرفوضة، ولا يوجد تنفيذ تداول أو تبديل لإعدادات المحرك الافتراضية.

## مختبر الأهداف والوقف المتحرك

trailing_lab.py ومهمة KDJ Trailing Exit Lab يختبران أهداف ربح ووقفًا متحركًا بالنسبة أو ATR، مع واجهة تقرير عربية تعمل دون إنترنت في lab.html داخل Artifacts. التفاصيل في reports/TRAILING_LAB_AR.md. النتيجة الحالية لا تحتوي إعدادًا مؤهلًا للتداول.

## Technical Strategy Engine (without KDJ)

Nine causal spot strategy families, regime/volume filters, 108 exit combinations, three validation windows and a reserved final test. Run **Actions → Technical Strategy Engine**, download the artifact and open `index.html`. Set `fetch_days=0` for stored data or `11..730` for public Bybit download. No exchange orders or machine-learning training. See [Arabic rules and limitations](TECHNICAL_ENGINE_AR.md).
