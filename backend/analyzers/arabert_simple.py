from backend.analyzers.base_tool import BaseTool


class AraBERTTool(BaseTool):
    name = "arabert"
    approach = "neural-bert"

    def analyze(self, text):
        from app.tools.arabert_tool import arabert_analyze

        return arabert_analyze(text)

    def is_loaded(self):
        from app.tools.arabert_tool import get_arabert_status_detail

        return get_arabert_status_detail().get("status") == "ok"


arabert_tool = AraBERTTool()
