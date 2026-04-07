import json
import sys
import io
import signal
import traceback
from datasets import load_from_disk


# ── Timeout helper ────────────────────────────────────────────────────────────
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timed out")


# ── Runner for stdin/stdout problems ─────────────────────────────────────────
def run_io_test(solution_code: str, stdin_input: str, timeout: int = 5) -> str | None:
    """Execute solution_code with stdin_input, return stdout or None on error."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        old_stdin  = sys.stdin
        old_stdout = sys.stdout
        sys.stdin  = io.StringIO(stdin_input)
        sys.stdout = io.StringIO()

        exec_globals = {}
        exec(solution_code, exec_globals)

        output = sys.stdout.getvalue()
        return output
    except TimeoutError:
        return None
    except Exception:
        traceback.print_exc()
        return None
    finally:
        signal.alarm(0)
        sys.stdin  = old_stdin
        sys.stdout = old_stdout


# ── Runner for function-call problems (fn_name present) ──────────────────────
def run_fn_test(solution_code: str, fn_name: str, inputs: list, timeout: int = 5):
    """Execute solution_code, call fn_name(*inputs), return result or None."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        exec_globals = {}
        exec(solution_code, exec_globals)
        fn = exec_globals[fn_name]
        result = fn(*inputs)
        return result
    except TimeoutError:
        return None
    except Exception:
        traceback.print_exc()
        return None
    finally:
        signal.alarm(0)


# ── Normalise output for comparison ──────────────────────────────────────────
def normalise(s: str) -> str:
    return s.strip().replace("\r\n", "\n")


# ── Evaluate all test cases ───────────────────────────────────────────────────
def evaluate(solution_code: str, input_output: dict) -> dict:
    inputs  = input_output["inputs"]
    outputs = input_output["outputs"]
    fn_name = input_output.get("fn_name")   # present for function-call problems

    results = []
    for inp, expected in zip(inputs, outputs):
        print(inp, expected)
        if fn_name:
            # inputs are already parsed as Python objects in fn_name problems
            actual = run_fn_test(solution_code, fn_name, inp if isinstance(inp, list) else [inp])
            passed = str(actual).strip() == str(expected).strip()
        else:
            actual = run_io_test(solution_code, inp)
            passed = actual is not None and normalise(actual) == normalise(expected)

        results.append({
            "input":    inp,
            "expected": expected,
            "actual":   actual,
            "passed":   passed,
        })

    n_passed = sum(r["passed"] for r in results)
    return {
        "passed":  n_passed,
        "total":   len(results),
        "reward":  n_passed / len(results) if results else 0.0,   # ← use as RL reward
        "details": results,
    }

# ── Load dataset ──────────────────────────────────────────────────────────────
ds = load_from_disk("/home/chanb/scratch/datasets/apps/data/apps_hint_sep/train")

for sample_i, sample in enumerate(ds):
    question    = sample["question"]
    input_output = json.loads(sample["test_cases"]) if sample["test_cases"] else None
    candidate_solution   = sample["answer"][0][10:-3]
    print(f"Sample {sample_i}:")
    print(input_output)
    print(candidate_solution)

    # ── Run ───────────────────────────────────────────────────────────────────────
    if input_output:
        report = evaluate(candidate_solution, input_output)
        print(f"Passed: {report['passed']} / {report['total']}")
        print(f"Reward: {report['reward']:.2f}")
        for i, r in enumerate(report["details"]):
            status = "✓" if r["passed"] else "✗"
            print(f"  [{status}] test {i+1}")
    else:
        print("No test cases available for this sample.")
