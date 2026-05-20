from backend.analyzers.base_tool import BaseTool

class MADAMIRATool(BaseTool):
    name = "madamira"
    approach = "java-http"
    
    def analyze(self, text):
        return {"tool":"madamira","status":"not_implemented","input":text,"word_count":0,"tokens":[]}
    
    def is_loaded(self): 
        return False

madamira_tool = MADAMIRATool()
