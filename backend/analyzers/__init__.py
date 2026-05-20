try:
    from backend.analyzers.udpipe_simple   import udpipe_tool
    from backend.analyzers.alkhalil_simple import alkhalil_tool
    from backend.analyzers.arabert_simple  import arabert_tool
    from backend.analyzers.madamira_simple import madamira_tool
    PARTNER_TOOLS = {
        "udpipe":   udpipe_tool,
        "alkhalil": alkhalil_tool,
        "arabert":  arabert_tool,
        "madamira": madamira_tool,
    }
except ImportError:
    PARTNER_TOOLS = {}

def get_all_partner_statuses():
    return {
        name: {"status": tool.get_status()}
        for name, tool in PARTNER_TOOLS.items()
    }
