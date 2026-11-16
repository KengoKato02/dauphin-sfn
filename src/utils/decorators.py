# ==============================================================================
# Title:       Decorators Utilities
# File:        decorators.py
# Author:      KATO Kengo
# Description: Simple Decorators setup for any decorator functions
# ==============================================================================

from functools import wraps

import sys
import traceback


def handle_exceptions(fn):
    """Log and re-raise exceptions, preserving the wrapped function's signature.

    This decorator and its own wrapper are bootstrap functions: applying the
    decorator to its implementation would recurse into itself.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            tb = sys.exc_info()[2]
            last_tb = traceback.extract_tb(tb)[-1]
            print('=================================================================')
            print(f"Exception caught in function: {fn.__name__}")
            print(f"Error: {e}")
            print()
            print(f"Triggered from File: {last_tb.filename}")
            print(f"Triggered from Function: {last_tb.name}")
            print(f"Trigger from Line: {last_tb.lineno}")
            print('=================================================================')
            raise
    return wrapper


@handle_exceptions
def log_arguments(fn):
    """Decorator to log function arguments."""
    @handle_exceptions
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"Calling '{fn.__name__}' with:")
        print(f"    args: {args}")
        print(f"    kwargs: {kwargs}")
        return fn(*args, **kwargs)
    return wrapper
