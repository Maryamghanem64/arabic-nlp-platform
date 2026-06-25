from backend.analyzers.base_tool import BaseTool


class AlKhalilTool(BaseTool):
    name = "alkhalil"
    approach = "java-bridge"

    def analyze(self, text):
        from app.tools.alkhalil_tool import alkhalil_analyze

        return alkhalil_analyze(text)

    def is_loaded(self):
        from app.tools.alkhalil_tool import get_alkhalil_status

        return get_alkhalil_status().get("status") == "ok"


alkhalil_tool = AlKhalilTool()
