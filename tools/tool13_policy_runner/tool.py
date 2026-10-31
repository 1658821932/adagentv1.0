"""Compatibility entry point for the mock; prefer mock.py or registry."""
from .mock import run

__all__ = ["run"]
