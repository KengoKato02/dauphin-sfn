# ==============================================================================
# Title:       Logger Utilities
# File:        utils/logger.py
# Author:      KATO Kengo
# Description: Logger functions to stringify Messages
# ==============================================================================

from collections.abc import Mapping
from src.utils.decorators import handle_exceptions


@handle_exceptions
def generic_logger_singleton(msg: str, val):
    """Generic logger to display concat between msg and val"""
    print(f"{msg}: ", val)


@handle_exceptions
def generic_logger(values: Mapping[str, object]) -> None:
    """Print each label and its corresponding value."""
    for label, value in values.items():
        print(f"{label}: {value}")


@handle_exceptions
def divider_logger():
    print("==================================================")
