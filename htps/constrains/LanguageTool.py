import language_tool_python


class LanguageTool():
    def __init__(self):
        self.lang_tool = language_tool_python.LanguageTool("en-US")
        self.grammar_error_cache = {}

    def get_errors(self, text, use_cache=False):
        if use_cache:
            if text not in self.grammar_error_cache:
                self.grammar_error_cache[text] = len(self.lang_tool.check(text))
            return self.grammar_error_cache[text]
        else:
            return len(self.lang_tool.check(text))
