# Copyright 2026 SDAR ALFWorld reproduction contributors.
# SPDX-License-Identifier: Apache-2.0
"""Select existing native environments for fixed-task evaluation only."""
import copy
import hashlib
import json
import os
from pathlib import Path


def fixed_validation_envs(base_env, manifest, env_num, group_n):
    rows = json.loads(Path(manifest).read_text(encoding='utf-8'))
    if env_num != len(rows) or group_n != 1:
        raise ValueError('Fixed validation requires one environment per manifest task')
    root = Path(os.environ['ALFWORLD_DATA']).resolve()
    allowed = {Path(p).resolve() for p in base_env.game_files}
    selected = []
    for row in rows:
        game = (root / row['task_id']).resolve()
        if not game.is_relative_to(root / 'json_2.1.1/valid_seen') or game not in allowed:
            raise ValueError('Task is outside the native seen validation pool')
        if hashlib.sha256(game.read_bytes()).hexdigest() != row['game_sha256']:
            raise ValueError('Validation game SHA mismatch')
        env = copy.copy(base_env)
        env.game_files, env.num_games = [str(game)], 1
        env.fixed_validation_seed = row['seed']
        selected.append(env)
    return selected
