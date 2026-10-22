"""Default numerical settings shared by the solver, runner and diagnostics."""
import json
from pathlib import Path

PROFILE_PATH = Path(__file__).with_name('solver_profile.json')


def load_profile(path):
    path = Path(path).resolve()
    visited = set()
    while path not in visited:
        visited.add(path)
        profile = json.loads(path.read_text(encoding='utf-8-sig'))
        if 'profile_file' not in profile:
            return profile
        path = (path.parent / profile['profile_file']).resolve()
    raise ValueError('Cyclic profile_file reference')


DEFAULT_PROFILE = load_profile(PROFILE_PATH)


def continuation_defaults():
    options = dict(DEFAULT_PROFILE['continuation'])
    options['h_init'] = options.pop('step')
    return options
