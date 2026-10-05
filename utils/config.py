"""Configuration loading, merging, validation, and immutable access helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

REQUIRED_CONFIG_KEYS = ("paths", "logging", "image", "device", "checkpoint")


def _repository_root() -> Path:
    """Find the repository root from this module's location."""
    module_path = Path(__file__).resolve()

    for candidate in module_path.parents:
        if (candidate / "project_explanation.md").exists() or (
            candidate / ".docs" / "project_explanation.md"
        ).exists():
            return candidate

    return module_path.parents[1]


def _resolve_config_path(path: str | Path, base_dir: Path) -> Path:
    """Resolve a configuration path relative to the repository when needed."""
    candidate = Path(path)

    if candidate.is_absolute():
        return candidate

    relative_to_config = base_dir / candidate

    if relative_to_config.exists():
        return relative_to_config

    return _repository_root() / candidate


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge an override mapping into a copy of a base mapping."""
    merged = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def _set_dotted_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value addressed by a dotted key, rejecting unknown paths."""
    parts = dotted_key.split(".")
    current = config

    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            raise KeyError(f"Cannot apply override; missing key path: {dotted_key}")

        current = current[part]

    if parts[-1] not in current:
        raise KeyError(f"Cannot apply override; missing key: {dotted_key}")

    current[parts[-1]] = value


def _convert_paths(value: Any, key_path: tuple[str, ...] = ()) -> Any:
    """Convert configured path values to absolute ``Path`` objects."""
    if isinstance(value, dict):
        return {
            key: _convert_paths(item, (*key_path, key)) for key, item in value.items()
        }

    if isinstance(value, list):
        return [_convert_paths(item, key_path) for item in value]

    if key_path and key_path[0] == "paths" and isinstance(value, str):
        return (_repository_root() / value).resolve()

    return value


def _plain_value(value: Any) -> Any:
    """Convert configuration objects and paths into YAML/JSON-safe values."""
    if isinstance(value, Config):
        return _plain_value(value.to_dict())

    if isinstance(value, dict):
        return {key: _plain_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_plain_value(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value


class Config(Mapping[str, Any]):
    """Immutable nested configuration with attribute and mapping access.

    Args:
        values: Configuration values to wrap.
        locked: Whether mutation is disabled initially.
    """

    def __init__(self, values: Mapping[str, Any], locked: bool = True) -> None:
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_locked", locked)

        for key, value in values.items():
            self._values[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls(value)

        if isinstance(value, list):
            return [cls._wrap(item) for item in value]

        return value

    def __getattr__(self, name: str) -> Any:
        values = self.__dict__.get("_values")
        if values is None:
            raise AttributeError(name)
        try:
            return values[name]
        except KeyError as error:
            raise AttributeError(f"Unknown configuration key: {name}") from error

    def __getstate__(self) -> dict[str, Any]:
        """Return a pickle-safe state for Windows DataLoader workers."""
        return {"values": self.to_dict(), "locked": self._locked}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a configuration received by a spawned worker."""
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_locked", bool(state["locked"]))
        for key, value in state["values"].items():
            self._values[key] = self._wrap(value)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        if self._locked:
            raise TypeError("Config is immutable; use cfg.unlocked() to mutate it")

        self._values[name] = self._wrap(value)

    def __setitem__(self, key: str, value: Any) -> None:
        self.__setattr__(key, value)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep plain-dictionary copy of the configuration."""
        return {key: _plain_value(value) for key, value in self._values.items()}

    @contextmanager
    def unlocked(self) -> Iterator[Config]:
        """Temporarily allow mutation of this configuration and its children."""
        previous_states = self._set_lock_state(False)

        try:
            yield self
        finally:
            self._restore_lock_state(previous_states)

    def _set_lock_state(self, locked: bool) -> list[tuple[Config, bool]]:
        states = [(self, self._locked)]
        object.__setattr__(self, "_locked", locked)

        for value in self._values.values():
            if isinstance(value, Config):
                states.extend(value._set_lock_state(locked))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Config):
                        states.extend(item._set_lock_state(locked))

        return states

    @staticmethod
    def _restore_lock_state(states: list[tuple[Config, bool]]) -> None:
        for config, locked in states:
            object.__setattr__(config, "_locked", locked)


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
    """Load, merge, validate, and resolve a YAML configuration.

    Args:
        path: YAML file to load.
        overrides: Optional dotted-key overrides for existing keys.

    Returns:
        The resolved immutable configuration.

    Raises:
        FileNotFoundError: If the requested configuration or base is missing.
        ValueError: If YAML is invalid or required sections are absent.
    """
    requested_path = _resolve_config_path(path, _repository_root())

    if not requested_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {requested_path}")

    try:
        loaded = yaml.safe_load(requested_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {requested_path}: {error}") from error

    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {requested_path}")

    default_base = _resolve_config_path("configs/base.yaml", _repository_root())
    base_reference = loaded.pop("_base_", "__default_base__")
    if base_reference == "__default_base__":
        base_references = (
            () if requested_path.resolve() == default_base.resolve() else ("configs/base.yaml",)
        )
    elif isinstance(base_reference, list):
        base_references = tuple(str(reference) for reference in base_reference)
    else:
        base_references = (str(base_reference),)

    merged: dict[str, Any] = {}
    for reference in base_references:
        base_path = _resolve_config_path(reference, requested_path.parent)
        base_config = load_config(base_path, overrides=None)
        merged = _deep_merge(merged, base_config.to_dict())
    merged = _deep_merge(merged, loaded)

    if overrides is not None:
        for dotted_key, value in overrides.items():
            _set_dotted_value(merged, dotted_key, value)

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in merged]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Configuration is missing required sections: {missing}")

    return Config(_convert_paths(merged))


def config_hash(cfg: Config | Mapping[str, Any]) -> str:
    """Return a stable short SHA-256 hash for a configuration."""
    values = cfg.to_dict() if isinstance(cfg, Config) else _plain_value(dict(cfg))
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()

    return hashlib.sha256(encoded).hexdigest()[:12]


def save_config(cfg: Config | Mapping[str, Any], path: str | Path) -> Path:
    """Write a resolved configuration snapshot to a YAML file.

    Args:
        cfg: Configuration to serialize.
        path: Destination YAML path.

    Returns:
        The resolved destination path.

    Raises:
        OSError: If the destination cannot be created or written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        destination.write_text(
            yaml.safe_dump(_plain_value(cfg), sort_keys=False), encoding="utf-8"
        )
    except OSError as error:
        raise OSError(
            f"Could not save configuration to {destination}: {error}"
        ) from error

    return destination
