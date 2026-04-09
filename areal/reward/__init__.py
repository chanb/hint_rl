import json
import multiprocessing
import re

from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from areal.utils import logging
from areal.utils.pytest_util import run_test

logger = logging.getLogger("RewardUtils")

VALID_REWARD_FN = ["clevr_count_70k", "geometry3k"]


def get_custom_reward_fn(path: str, **kwargs):
    if "clevr_count_70k" in path:
        from .clevr_count_70k import clevr_count_70k_reward_fn

        return clevr_count_70k_reward_fn
    elif "geometry3k" in path:
        from .geometry3k import geometry3k_reward_fn

        return geometry3k_reward_fn
    else:
        raise ValueError(
            f"Reward function {path} is not supported. "
            f"Supported reward functions are: {VALID_REWARD_FN}. "
        )


class MathVerifyWorker:
    """Thin wrapper over math_verify with configurable extraction/precision.

    Args:
        try_extract_without_anchor: When False, only answers with explicit anchors
            (e.g., "answer = 1", "final answer = 1") are matched. When True,
            any numeric string in the text may be extracted.
        precision: Number of significant digits that must match.

    Notes:
        Tune these knobs based on dataset format and model output style.
    """

    def __init__(self, try_extract_without_anchor=True, precision: int = 6):
        self.verify_func = math_metric(
            gold_extraction_target=(
                ExprExtractionConfig(
                    try_extract_without_anchor=try_extract_without_anchor
                ),
                LatexExtractionConfig(),
            ),
            pred_extraction_target=(
                ExprExtractionConfig(
                    try_extract_without_anchor=try_extract_without_anchor
                ),
                LatexExtractionConfig(),
            ),
            precision=precision,
        )

    def verify(self, response: str, ground_truth: str) -> float:
        # ground_truth_parsable = "\\boxed{" + ground_truth + "}"
        try:
            ret_score, _ = self.verify_func([ground_truth], [response])
            return float(ret_score)
        except Exception:
            logger.warning(
                f"Exception in MathVerifyWorker.verify for response={response} and ground_truth={ground_truth}",
                exc_info=True,
            )
            return 0.0

class CodeVerifyWorker:
    """A Python code verifier
    """

    def __init__(self, debug=False):
        def check_correctness(test_cases, generation):
            """Check correctness of code generation with a global timeout.
            The global timeout is to catch some extreme/rare cases not handled by the timeouts
            inside `run_test`"""
            def _temp_run(test_cases, generation, debug, result):
                result.append(run_test(test_cases, test=generation, debug=debug))

            pattern = r"```python\s*\r?\n(.*?)\r?\n```"
            codes = [block.strip() for block in re.findall(pattern, generation, re.DOTALL)]

            if len(codes) == 0:
                logger.info("no code found")
                return [False]

            generation = codes[0]

            manager = multiprocessing.Manager()
            result = manager.list()
            in_outs = json.loads(test_cases)
            p = multiprocessing.Process(target=_temp_run, args=(in_outs, generation, debug, result))
            p.start()
            p.join()
            if p.is_alive():
                p.kill()
            if not result:
                # consider that all tests failed
                result = [[-1 for i in range(len(in_outs["inputs"]))]]
                logger.debug(f"global timeout")
            return result[0]
        self.verify_func = check_correctness

    def verify(self, response: str, test_cases: str) -> float:
        try:
            ret_score = self.verify_func(test_cases, response)
            return float(all(ret_score))
        except Exception:
            logger.warning(
                f"Exception in CodeVerifyWorker.verify for response={response} and test_cases={test_cases}",
                exc_info=True,
            )
            return 0.0


_MATH_VERIFY_WORKER: MathVerifyWorker | None = None
_CODE_VERIFY_WORKER: CodeVerifyWorker | None = None


def get_math_verify_worker() -> MathVerifyWorker:
    global _MATH_VERIFY_WORKER
    if _MATH_VERIFY_WORKER is None:
        _MATH_VERIFY_WORKER = MathVerifyWorker()
    return _MATH_VERIFY_WORKER

def get_code_verify_worker() -> CodeVerifyWorker:
    global _CODE_VERIFY_WORKER
    if _CODE_VERIFY_WORKER is None:
        _CODE_VERIFY_WORKER = CodeVerifyWorker()
    return _CODE_VERIFY_WORKER


__all__ = [
    "VALID_REWARD_FN",
    "get_custom_reward_fn",
    "MathVerifyWorker",
    "CodeVerifyWorker",
    "get_math_verify_worker",
    "get_code_verify_worker",
    "gsm8k_reward_fn",
    "geometry3k_reward_fn",
    "clevr_count_70k_reward_fn",
    "opencode_reward_fn",
]


_LAZY_IMPORTS = {
    "gsm8k_reward_fn": "areal.reward.gsm8k",
    "geometry3k_reward_fn": "areal.reward.geometry3k",
    "clevr_count_70k_reward_fn": "areal.reward.clevr_count_70k",
    "opencode_reward_fn": "areal.reward.opencode",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
