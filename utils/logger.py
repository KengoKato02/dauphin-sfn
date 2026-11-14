# ==============================================================================
# Title:       Logger Utilities
# File:        utils/logger.py
# Author:      KATO Kengo
# Description: Logger functions to stringify Messages
# ==============================================================================

from decorators import handle_exceptions


@handle_exceptions
def generic_logger_singleton(msg: str, val):
    """Generic logger to display concat between msg and val"""
    print(f"{msg}: ", val)


@handle_exceptions
def generic_logger_tuple(vals: Tuple):
    """Loop over tuple and smart construct log messages"""
    for val in vals:
        print(f"")
