from .types import QueryLabel, ClassificationResult
from typing import Union

def action_for(label_or_result: Union[QueryLabel, ClassificationResult]) -> str:
    # Handle both QueryLabel and ClassificationResult for backward compatibility
    if isinstance(label_or_result, ClassificationResult):
        label = label_or_result.label
    else:
        label = label_or_result
    
    actions = {
        QueryLabel.RELEVANT: "Proceed to LLM answer.",
        QueryLabel.IRRELEVANT: "Reject or ignore.",
        QueryLabel.VAGUE: "Ask user to clarify.",
        QueryLabel.UNKNOWN: "Manual review.",
    }
    return actions.get(label, "Manual review.")