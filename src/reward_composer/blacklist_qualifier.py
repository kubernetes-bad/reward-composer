import json

from .qualifiers import Qualifier, QualifierInput


class BlacklistQualifier(Qualifier):
    """Qualifier that checks if completion does not have any blacklisted words/phrases."""

    def __init__(self, blacklist_path: str, case_sensitive: bool = False):
        super().__init__(name="blacklist_qualifier")
        with open(blacklist_path, 'r') as f:
            self.blacklist = json.load(f)
        self.case_sensitive = case_sensitive

    def __call__(self, completion: str, context: QualifierInput = None) -> bool:
        if self.case_sensitive:
            return not any(word in completion for word in self.blacklist)

        completion = completion.lower()
        blacklist_lower = [word.lower() for word in self.blacklist]
        return not any(word in completion for word in blacklist_lower)
