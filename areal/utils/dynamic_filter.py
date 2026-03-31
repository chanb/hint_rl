from typing import Any, Dict, Callable

def filter_always_fail_pass(x: Dict[str, Any]):
    return 0 < x["rewards"].mean() < 1
