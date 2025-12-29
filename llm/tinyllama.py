import subprocess

class TinyLlamaLLM:
    """
    Local TinyLlama LLM wrapper using Ollama.
    """

    def __init__(self, model_name="tinyllama"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Generate a response from TinyLlama using Ollama CLI.
        """
        result = subprocess.run(
            ["ollama", "run", self.model_name],
            input=prompt,
            text=True,
            capture_output=True
        )

        return result.stdout.strip()
