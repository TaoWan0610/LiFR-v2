"""Shared YAML configuration loading and command-line overrides."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def load_config(path: str | Path) -> dict:
    """Read a complete YAML configuration template."""
    try:
        import yaml
    except ImportError as error:
        try:
            from invoke.vendor import yaml
        except ImportError:
            raise ImportError("Reading configuration files requires PyYAML: pip install pyyaml") from error

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")
    if not isinstance(config, dict):
        raise TypeError(f"The top-level configuration must be a dictionary: {path}")
    return config


def _parse_override_value(text: str) -> Any:
    if text == "":
        return ""
    try:
        import yaml
    except ImportError as error:
        try:
            from invoke.vendor import yaml
        except ImportError:
            raise ImportError("Parsing configuration overrides requires PyYAML: pip install pyyaml") from error
    return yaml.safe_load(text)


def apply_overrides(config: Mapping[str, Any], overrides: list[str] | tuple[str, ...]) -> dict:
    """Apply command-line overrides in ``A.B=value`` format and return a new configuration dictionary."""
    updated = deepcopy(dict(config))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Configuration overrides must use KEY=VALUE format; got: {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        keys = dotted_key.split(".")
        if not dotted_key or any(not key for key in keys):
            raise ValueError(f"Invalid configuration override key: {dotted_key!r}")

        current = updated
        for key in keys[:-1]:
            value = current.get(key)
            if value is None:
                value = {}
                current[key] = value
            if not isinstance(value, dict):
                raise TypeError(f"Cannot override {dotted_key!r}: {key!r} is not a dictionary")
            current = value
        current[keys[-1]] = _parse_override_value(raw_value)
    return updated


def config_duration_ms(config: Mapping[str, Any], command_line_value: int | None) -> int:
    """Use the command-line duration when provided; otherwise read EXPERIMENT.DURATION_MS."""
    if command_line_value is not None:
        duration = int(command_line_value)
    else:
        duration = int(config.get("EXPERIMENT", {}).get("DURATION_MS", 0))
    if duration < 0:
        raise ValueError(f"duration must be a nonnegative integer; got: {duration}")
    return duration


__all__ = ["apply_overrides", "config_duration_ms", "load_config"]
