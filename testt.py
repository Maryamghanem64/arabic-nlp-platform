from sinatools.morphology import MorphAnalyzer

def analyze_arabic_sentence(text):
    try:
        # 1. تهيئة المحلل الصرفي
        # سيقوم هذا السطر بتحميل القواميس من AppData
        analyzer = MorphAnalyzer()
        
        # 2. تحليل النص
        # الدالة analyze ترجع قائمة (list) تحتوي على تفاصيل كل كلمة
        results = analyzer.analyze(text)
        
        print(f"التحليل الصرفي للجملة: '{text}'\n")
        print(f"{'الكلمة':<12} | {'الوسم (POS)':<15} | {'الجذر (Root)':<10}")
        print("-" * 45)
        
        for word_data in results:
            word = word_data['word']
            # POS Tag يمثل نوع الكلمة (فعل، اسم، حرف...)
            pos = word_data['pos'] 
            root = word_data.get('root', 'N/A')
            
            print(f"{word:<12} | {pos:<15} | {root:<10}")
            
    except Exception as e:
        print(f"حدث خطأ أثناء التحليل: {e}")
        print("تأكد من تشغيل سكريبت تحميل الملفات أولاً.")

if __name__ == "__main__":
    sentence = 'ذهب الولد إلى المدرسة'
    analyze_arabic_sentence(sentence)