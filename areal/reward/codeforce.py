from areal.utils import logging

from . import get_code_verify_worker

logger = logging.getLogger("CodeforceReward")


def codeforce_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    try:
        print(kwargs.keys())
        assert 0
        worker = get_code_verify_worker()
        return worker.verify(str(completions), str(answer))
    except Exception:
        logger.warning("Exception in codeforce_reward_fn", exc_info=True)
        return 0.0
