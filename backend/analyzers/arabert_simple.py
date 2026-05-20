from backend.analyzers.base_tool import BaseTool

class AraBERTTool(BaseTool):
    name = "arabert"
    approach = "neural-bert"
    
    def analyze(self, text):
        return {"tool":"arabert","status":"not_implemented","input":text,"word_count":0,"tokens":[]}
    
    def is_loaded(self): 
        return False

arabert_tool = AraBERTTool()
