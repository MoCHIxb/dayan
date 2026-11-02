"""Optional explicit training budget; legacy epoch configurations stay unchanged."""
from pathlib import Path


def budget_arguments(config):
    if "max_steps" not in config:
        return {}
    for key in ("max_steps", "eval_steps", "save_steps"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["eval_steps"] != config["save_steps"]:
        raise ValueError("Evaluation and checkpoint intervals must match")
    if config["max_steps"] % config["eval_steps"]:
        raise ValueError("The final update must also be evaluated and saved")
    if config.get("eval_max_samples") != 1500:
        raise ValueError("Explicit-budget experiments require the complete 1500-record dev")
    return {"max_steps": config["max_steps"], "eval_strategy": "steps",
            "save_strategy": "steps", "eval_steps": config["eval_steps"],
            "save_steps": config["save_steps"], "include_num_input_tokens_seen": True}


def protect_output(config, resume):
    """New protocol runs cannot silently overwrite or resume a different config."""
    if not config.get("protocol_id"):
        return
    import json
    output = Path(config["output_dir"])
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Nonempty experiment directory: {output}")
        return
    checkpoint = Path(resume).resolve()
    if checkpoint.parent != (output / "checkpoints").resolve() or not checkpoint.is_dir():
        raise ValueError("Resume checkpoint must belong to this run")
    saved = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    if saved != config:
        raise ValueError("Resume config differs from the frozen run config")
