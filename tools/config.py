"""Render portable configuration templates without importing model dependencies."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIABLE = re.compile(r'\$\{([A-Z_][A-Z_0-9]*)\}')


def read_config(path):
    path = Path(path)
    if path.suffix == '.json':
        return json.loads(path.read_text())
    import yaml
    return yaml.safe_load(path.read_text())


def roots(data_root=None, weights_root=None, output_root=None):
    values = {'REPO_ROOT': ROOT,
              'DATA_ROOT': data_root or os.environ.get('LIFR_DATA_ROOT', ROOT / 'data'),
              'WEIGHTS_ROOT': weights_root or os.environ.get('LIFR_WEIGHTS_ROOT', ROOT / 'weights'),
              'OUTPUT_ROOT': output_root or os.environ.get('LIFR_OUTPUT_ROOT', ROOT / 'outputs')}
    return {k: str(Path(v).expanduser().resolve()) for k, v in values.items()}


def expand(value, variables):
    if isinstance(value, dict):
        return {k: expand(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, variables) for v in value]
    if isinstance(value, str):
        def substitute(match):
            name = match.group(1)
            if name not in variables:
                raise ValueError(f'Undefined configuration variable: {name}')
            return variables[name]
        return VARIABLE.sub(substitute, value)
    return value


def render_tree(destination, variables):
    """Keep sibling eval configs and nested teacher configurations together."""
    destination = Path(destination)
    config_root = ROOT / 'configs'

    def relocate(value):
        if isinstance(value, dict): return {k: relocate(v) for k,v in value.items()}
        if isinstance(value, list): return [relocate(v) for v in value]
        if isinstance(value, str) and value.startswith(str(config_root) + '/'):
            return str(destination / Path(value).relative_to(config_root))
        return value

    for source in config_root.rglob('*'):
        if source.suffix not in {'.json', '.yaml', '.yml'}: continue
        target = destination / source.relative_to(config_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        config = relocate(expand(read_config(source), variables))
        if target.suffix == '.json':
            target.write_text(json.dumps(config, indent=2) + '\n')
        else:
            import yaml
            target.write_text(yaml.safe_dump(config, sort_keys=False))
    return destination

