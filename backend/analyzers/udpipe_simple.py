from backend.analyzers.base_tool import BaseTool

class UDPipeTool(BaseTool):
    name = "udpipe"
    approach = "rest-api"
    
    def analyze(self, text):
        return {"tool":"udpipe","status":"not_implemented","input":text,"word_count":0,"tokens":[]}
    
    def is_loaded(self): 
        return False

udpipe_tool = UDPipeTool()
