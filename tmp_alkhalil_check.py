from app.tools.alkhalil_tool import alkhalil_analyze
import json

samples = ['كتب الطالب الدرس', 'ذهب الرجل إلى المدرسة', 'الكتاب يقرأه الطلاب']
for s in samples:
    res = alkhalil_analyze(s)
    print('SAMPLE', s)
    print(json.dumps({'status': res.get('status'), 'tokens': len(res.get('tokens', [])), 'sample': res.get('tokens', [])[:3]}, ensure_ascii=False, indent=2))
    print('---')
