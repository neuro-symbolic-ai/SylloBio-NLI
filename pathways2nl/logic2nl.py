import json
from dynaconf import settings


class Logic2NLTranslator:
    def __init__(self):
        with open(settings["sttm_template_file_path"]) as templ_file:
            self.templates = json.load(templ_file)

    def translate(self, formula: str, **kwargs) -> str:
        if (formula not in self.templates):
            raise NotImplementedError("Requested formula has not been translated yet")

        return self.templates[formula].format(**kwargs)
