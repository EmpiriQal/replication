"""Chat replies from a Gemini model on Vertex AI (Google Cloud), through the google-genai package, signed in with gcloud's application-default credentials. The Google Cloud project is read from GOOGLE_CLOUD_PROJECT and the location from GOOGLE_CLOUD_LOCATION (default "global"). Gemini returns no log-probabilities, so only the text is read."""

import os


class GeminiError(Exception):
    """A request failed or returned something unexpected."""


class GeminiClient:
    """Each reply is returned as a dict: the text, the finish reason, and the token counts (prompt, reply, and reasoning). Safe to share between threads."""

    def __init__(self, model: str, client=None):
        self.model = model
        if client is None:
            from google import genai
            from google.genai import types
            if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
                raise GeminiError("set GOOGLE_CLOUD_PROJECT to a Google Cloud project with Vertex AI enabled")
            # retries cover rate limits (429) and server errors, with exponential back-off
            retry = types.HttpRetryOptions(attempts=8, initial_delay=2, max_delay=120, http_status_codes=[408, 429, 500, 502, 503, 504])
            client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"), http_options=types.HttpOptions(timeout=600_000, retry_options=retry))
        self.client = client

    def ask(self, user: str, temperature: float | None, max_tokens: int, reasoning_budget: int | None = None) -> dict:
        """One reply to a single user message. max_tokens caps reasoning and answer together; reasoning_budget, if given, caps the reasoning in tokens."""
        from google.genai import types
        thinking = types.ThinkingConfig(thinking_budget=reasoning_budget) if reasoning_budget is not None else None
        try:
            r = self.client.models.generate_content(model=self.model, contents=user, config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens, thinking_config=thinking))
        except Exception as e:  # google-genai raises its own errors and httpx errors
            raise GeminiError(f"{self.model}: request failed: {e}") from None
        if not r.candidates:
            raise GeminiError(f"{self.model}: no reply (prompt feedback: {r.prompt_feedback})")
        c, u = r.candidates[0], r.usage_metadata
        text = "".join(p.text for p in (c.content.parts if c.content and c.content.parts else []) if p.text and not p.thought)
        if not text and not (u and u.thoughts_token_count):
            # finish reason STOP with no text and no reasoning: treated as a failed request, so it is sent again
            raise GeminiError(f"{self.model}: empty reply (finish reason {c.finish_reason})")
        return {
            "text": text,
            "finish_reason": str(c.finish_reason.name if c.finish_reason else None),
            "prompt_tokens": u.prompt_token_count if u else None,
            "reply_tokens": u.candidates_token_count if u else None,
            "reasoning_tokens": (u.thoughts_token_count or 0) if u else None,
        }
