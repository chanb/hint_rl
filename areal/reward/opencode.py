from areal.utils import logging

from . import get_code_verify_worker

logger = logging.getLogger("OpenCodeReward")


def opencode_reward_fn(
    prompt, completions, prompt_ids, completion_ids, test_cases, **kwargs
) -> float:
    try:
        worker = get_code_verify_worker()
        return worker.verify(str(completions), test_cases)
    except Exception:
        logger.warning("Exception in opencode_reward_fn", exc_info=True)
        return 0.0
