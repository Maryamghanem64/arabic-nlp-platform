from backend.analyzers.base_tool import BaseTool

class AlKhalilTool(BaseTool):
    name = "alkhalil"
    approach = "java-jar"
    
    def analyze(self, text):
        return {"tool":"alkhalil","status":"not_implemented","input":text,"word_count":0,"tokens":[]}
    
    def is_loaded(self): 
        return False

alkhalil_tool = AlKhalilTool()
