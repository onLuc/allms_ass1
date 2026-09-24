import argparse
import datetime as dt
import json
import re
import urllib.error
import urllib.parse
import urllib.request

from experiments.common import RESULTS, append_event, prepare, write_json


MODELS = (
    "allenai/OLMo-1B",
    "HuggingFaceTB/SmolLM2-360M",
    "Qwen/Qwen2.5-0.5B",
)


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "allms-assignment-experiments/1.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read().decode("utf-8")


def config_value(config, *keys):
    for key in keys:
        if config.get(key) is not None:
            return config[key]
    return None


def card_lines(card, pattern, limit=4):
    matches = []
    lines = card.splitlines()
    for index, line in enumerate(lines):
        normalized = re.sub(r"\s+", " ", line).strip(" -*#|`")
        if re.fullmatch(r"(?:training\s+)?(?:data|datasets?|tokens?|steps?|pretraining)(?:\s+(?:data|sources?))?", normalized, flags=re.IGNORECASE):
            for following in lines[index + 1:index + 4]:
                excerpt = re.sub(r"\s+", " ", following).strip(" -*#|`")
                if excerpt and not excerpt.startswith("<"):
                    normalized = f"{normalized}: {excerpt}"
                    break
        if normalized and re.search(pattern, normalized, flags=re.IGNORECASE):
            if normalized not in matches:
                matches.append(normalized[:320])
        if len(matches) == limit:
            break
    return matches


def parameter_count(info, config, card):
    weights = info.get("safetensors", {}).get("parameters", {})
    if weights:
        return sum(value for value in weights.values() if isinstance(value, int))
    count = config_value(config, "num_parameters", "n_params", "parameter_count")
    if count:
        return count
    hits = card_lines(card, r"\b(parameters|params)\b", 2)
    return hits or None


def model_record(model_id):
    encoded_id = urllib.parse.quote(model_id, safe="/")
    query = urllib.parse.urlencode([("expand[]", "safetensors"), ("expand[]", "cardData"), ("expand[]", "config")])
    api_url = "https://huggingface.co/api/models/" + encoded_id + "?" + query
    api = json.loads(fetch(api_url))
    config_url = f"https://huggingface.co/{encoded_id}/raw/main/config.json"
    card_url = f"https://huggingface.co/{encoded_id}/raw/main/README.md"
    try:
        config = json.loads(fetch(config_url))
        config_source = config_url
    except (urllib.error.HTTPError, json.JSONDecodeError):
        config = api.get("config", {})
        config_source = api_url + " (config fields returned by Hub API)"
    try:
        card = fetch(card_url)
    except urllib.error.HTTPError:
        card = ""
    data = card_lines(card, r"(data|dataset|corpus|pretrain|training mixture|trained on)", 5)
    training = card_lines(card, r"(training tokens|tokens seen|trained on|training steps|optimizer steps|steps)", 5)
    references = card_lines(card, r"(arxiv|doi|technical report|paper|citation)", 5)
    parameter_source = api_url if api.get("safetensors", {}).get("parameters") else config_source if config_value(config, "num_parameters", "n_params", "parameter_count") else card_url
    return {
        "model_id": model_id,
        "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "layers": config_value(config, "num_hidden_layers", "n_layer", "num_layers"),
        "hidden_dimension": config_value(config, "hidden_size", "n_embd", "d_model"),
        "attention_heads": config_value(config, "num_attention_heads", "n_head", "num_heads"),
        "kv_heads": config_value(config, "num_key_value_heads", "num_kv_heads"),
        "trainable_parameters": parameter_count(api, config, card),
        "training_tokens_or_steps_evidence": training or ["Not stated in the model card excerpt; consult the linked training paper."],
        "training_data_evidence": data or ["Not stated in the model card excerpt; consult the linked training paper."],
        "related_paper_evidence": references,
        "field_sources": {
            "layers_hidden_dimension_attention_heads": config_source,
            "trainable_parameters": parameter_source,
            "training_tokens_or_steps": card_url,
            "training_data_sources": card_url,
        },
        "sources": {
            "model_card": f"https://huggingface.co/{encoded_id}",
            "config": config_source,
            "hub_api_parameter_metadata": api_url,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    prepare()
    append_event({"event": "start", "stage": "task_0", "models": args.models})
    records = [model_record(model_id) for model_id in args.models]
    output = write_json(RESULTS / "task_0_model_hub.json", records)
    rows = [
        "| Model | Tokens or steps evidence | Layers | Hidden dim | Heads | Parameters | Training data evidence | Sources |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for item in records:
        fmt = lambda value: (",".join(str(part) for part in value) if isinstance(value, list) else str(value or "not reported")).replace("|", "\\|")
        sources = ", ".join(f"[{name}]({url})" for name, url in item["sources"].items())
        rows.append(
            f"| {item['model_id']} | {fmt(item['training_tokens_or_steps_evidence'])} | {fmt(item['layers'])} | "
            f"{fmt(item['hidden_dimension'])} | {fmt(item['attention_heads'])} | {fmt(item['trainable_parameters'])} | "
            f"{fmt(item['training_data_evidence'])} | {sources} |"
        )
    (RESULTS / "task_0_model_hub.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {output} and {RESULTS / 'task_0_model_hub.md'}")
    append_event({"event": "success", "stage": "task_0", "models": args.models})


if __name__ == "__main__":
    main()
