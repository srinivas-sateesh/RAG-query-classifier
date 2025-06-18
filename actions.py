from .types import QueryLabel

def action_for(label: QueryLabel) -> str:
    actions = {
        QueryLabel.RELEVANT: "Proceed to LLM answer.",
        QueryLabel.IRRELEVANT: "Reject or ignore.",
        QueryLabel.VAGUE: "Ask user to clarify.",
        QueryLabel.UNKNOWN: "Manual review.",
    }
    return actions.get(label, "Manual review.")