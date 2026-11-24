"""The chat templates as rendered for the chat models: the prompt of one claim (row 239, Woltin et al., 2010, with the original finding) wrapped by each model's own chat template, as vLLM's chat interface does (add_generation_prompt true; enable_thinking false or true where the template has it). Written to chat_templates/<name>.txt.

Reads each model's tokenizer from the local Hugging Face cache (HF_HOME, or the default cache) and sends nothing to any server. Needs the transformers package, which the analyses do not use:

    uv run --with transformers --with jinja2 python render_templates.py
"""

import tomllib

from transformers import AutoTokenizer

import score as S

ROW = 239
OUT = S.ROOT / "chat_templates"
# name in models.toml: the enable_thinking values rendered (None: the template has no such switch)
THINKING = {"qwen27": (False, True), "qwen08c": (False,), "qwen2c": (False,), "qwen4c": (False,), "qwen9c": (False,), "qwen30i": (None,), "olmo7i": (None,), "olmo32i": (None,), "olmo32t": (None,)}


def main() -> None:
    models = tomllib.loads((S.ROOT / "models.toml").read_text())["models"]
    rec = S.load_data(S.ROOT / S.CONFIG["data_file"])[ROW - 1]
    messages = [{"role": "user", "content": S.qwen_prompt(rec, "with_original_finding")}]
    OUT.mkdir(exist_ok=True)
    for name, settings in THINKING.items():
        tok = AutoTokenizer.from_pretrained(models[name]["model"])
        parts = []
        for th in settings:
            kw = {} if th is None else {"enable_thinking": th}
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
            label = "" if th is None else f"enable_thinking={th}"
            parts.append(f"===== {models[name]['model']}{', ' + label if label else ''} =====\n{text}")
        (OUT / f"{name}.txt").write_text("\n".join(parts), encoding="utf-8")
        print(f"wrote chat_templates/{name}.txt")


if __name__ == "__main__":
    main()
