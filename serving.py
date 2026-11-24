"""The local models listed in models.toml: each one's model ID, URL and kind. The models must already be served (see the `serve` command of each entry); nothing here starts or stops a server."""

import tomllib
from pathlib import Path

import httpx

FILE = Path(__file__).resolve().parent / "models.toml"


class ServingError(Exception):
    pass


def _models() -> dict:
    return tomllib.loads(FILE.read_text(encoding="utf-8"))["models"]


def entry(name: str) -> dict:
    models = _models()
    if name not in models:
        raise ServingError(f"{name!r} is not in models.toml (known: {', '.join(models)})")
    return models[name]


def endpoint(name: str) -> tuple[str, str]:
    """The OpenAI-compatible base URL (ending in /v1) and the model ID."""
    e = entry(name)
    return e["url"], e["model"]


def kind(name: str) -> str:
    """"chat" or "base"."""
    return entry(name)["kind"]


def is_ready(name: str) -> bool:
    """True if the server answers its health check."""
    url = endpoint(name)[0].removesuffix("/v1")
    try:
        return httpx.get(f"{url}/health", timeout=10).status_code == 200
    except httpx.HTTPError:
        return False


def require(name: str) -> None:
    if not is_ready(name):
        url, model = endpoint(name)
        raise ServingError(f"{name} ({model}) is not answering at {url}; start it with the serve command in models.toml")
