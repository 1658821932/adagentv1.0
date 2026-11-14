"""Manage the one shared reasoning API. Never accept or print credential values."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api_config

RESERVED_NAMES = frozenset({
    "retrieval", "active_profile", "reasoning_effort", "tool_profiles", "tool_efforts",
})


def _profile(config, name):
    if not isinstance(name, str) or name in RESERVED_NAMES or not isinstance(config.get(name), dict):
        raise ValueError("Select a saved reasoning profile; retrieval is managed separately")
    profile = config[name]
    api_config.validate_llm_profile(profile)
    return profile


def _summary(config, name):
    profile = _profile(config, name)
    return {
        "name": name,
        "active": name == config.get("active_profile", "kimi"),
        "model": profile["model"],
        "base_url": profile["base_url"],
        "api_key_env": profile["api_key_env"],
        "key_configured": bool(os.environ.get(profile["api_key_env"], "").strip()),
    }


def _save(path, config):
    # Save only public settings, atomically, preserving the config permissions.
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=api_config.API_CONFIG_PATH,
                        help="Public API config file (default: configs/api.json)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List saved profiles without showing keys")
    commands.add_parser("show", help="Show the current shared profile without its key")
    use = commands.add_parser("use", help="Switch all reasoning agents for the next run")
    use.add_argument("name")
    add = commands.add_parser("add", help="Save a profile using a key's environment variable name")
    add.add_argument("name")
    add.add_argument("--base-url", required=True)
    add.add_argument("--model", required=True)
    add.add_argument("--key-env", required=True, help="Variable name such as API_KEY_1; never the key itself")
    add.add_argument("--timeout", type=int, default=240)
    add.add_argument("--max-tokens", type=int, default=8192)
    add.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"))
    add.add_argument("--image-tool-choice", choices=("auto", "required"),
                     help="Explicit image+tool transport capability; local native-call validation always applies")
    add.add_argument('--required-tool-thinking', choices=('preserve','disabled'),
                     help='Explicit provider capability for forced tool calls; preserves native tool restrictions')
    add.add_argument('--request-dialect', choices=('thinking','qwen','kimi'))
    add.add_argument('--thinking-mode', choices=('enabled','disabled'))
    add.add_argument('--tool-choice-policy', choices=('preserve','auto'))
    args = parser.parse_args(argv)

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("API configuration must be an object")
        if args.command == "add":
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
                raise ValueError("Profile names may contain letters, digits, dots, dashes and underscores")
            if args.name in config or args.name in RESERVED_NAMES:
                raise ValueError("Profile name already exists or is reserved; choose a new name")
            profile = dict(base_url=args.base_url, model=args.model,
                           api_key_env=args.key_env, timeout=args.timeout, max_tokens=args.max_tokens)
            if args.reasoning_effort:
                profile["reasoning_effort"] = args.reasoning_effort
            if args.image_tool_choice:
                profile['image_tool_choice'] = args.image_tool_choice
            if args.required_tool_thinking:
                profile['required_tool_thinking'] = args.required_tool_thinking
            for field in ('request_dialect','thinking_mode','tool_choice_policy'):
                if getattr(args,field):
                    profile[field] = getattr(args,field)
            api_config.validate_llm_profile(profile)
            config[args.name] = profile
            _save(args.config, config)
            print("Saved profile. Use 'use NAME' before starting a new run.")
            return 0

        api_config.load_credentials()
        if args.command == "list":
            rows = [_summary(config, name) for name, profile in config.items()
                    if name not in RESERVED_NAMES and isinstance(profile, dict)]
        elif args.command == "show":
            rows = _summary(config, config.get("active_profile", "kimi"))
        else:
            profile = _profile(config, args.name)
            if not os.environ.get(profile["api_key_env"], "").strip():
                raise ValueError("Missing credential variable: " + profile["api_key_env"] + "; add it to .env first")
            config["active_profile"] = args.name
            _save(args.config, config)
            rows = _summary(config, args.name)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        if args.command == "use":
            print("Selected for new runs. Restart existing agents before using the new profile.")
        return 0
    except (OSError, ValueError, KeyError) as error:
        # Configuration errors are controlled messages; never dump profiles,
        # request headers, environment mappings or arbitrary exception strings.
        message = str(error) if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError) else type(error).__name__
        print("API profile error: " + message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
