from jinja2 import Template, StrictUndefined


class Prompt:
    """A prompt for a language model. This is a wrapper around a Jinja2 template in order to make it serializable for jsonargparse."""

    def __init__(self, template: str):
        self.template = template

    def __call__(self, **kwargs):
        for key, value in kwargs.items():
            if value is None:
                raise ValueError(f"Prompt argument '{key}' cannot be None.")
        return Template(self.template, undefined=StrictUndefined).render(**kwargs)


class ChatPrompt(Prompt):
    """A multi-role prompt for a language model using apply_chat_template."""

    def __init__(
        self,
        system: str = None,
        user: str = None,
        assistant: str = None,
        gcp: bool = False,
    ):
        self.system = system
        self.user = user
        self.assistant = assistant
        self.gcp = gcp

    @property
    def system_template(self):
        return Prompt(self.system) if self.system else None

    @property
    def user_template(self):
        return Prompt(self.user) if self.user else None

    @property
    def assistant_template(self):
        return Prompt(self.assistant) if self.assistant else None

    def __call__(self, **kwargs):
        if self.gcp:
            return self._render_gcp(**kwargs)
        return self._render_standard(**kwargs)

    def _render_standard(self, **kwargs):
        """Render messages in HuggingFace chat template format (role/content)."""
        messages = []
        if self.system_template:
            messages.append(
                {"role": "system", "content": self.system_template(**kwargs)}
            )
        if self.user_template:
            messages.append({"role": "user", "content": self.user_template(**kwargs)})
        if self.assistant_template:
            messages.append(
                {"role": "assistant", "content": self.assistant_template(**kwargs)}
            )
        return messages

    def _render_gcp(self, **kwargs):
        """Render messages in GCP GenAI format (role/parts, model instead of assistant).

        System instructions are included with role="system" so that GCPLLM
        can extract them and pass via config.system_instruction.
        """
        messages = []
        if self.system_template:
            messages.append(
                {"role": "system", "parts": [{"text": self.system_template(**kwargs)}]}
            )
        if self.user_template:
            messages.append(
                {"role": "user", "parts": [{"text": self.user_template(**kwargs)}]}
            )
        if self.assistant_template:
            messages.append(
                {
                    "role": "model",
                    "parts": [{"text": self.assistant_template(**kwargs)}],
                }
            )
        return messages
