import io
import json
import sys
import traceback
from dataclasses import dataclass, field
from datasets import load_from_disk


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    index:    int
    passed:   bool
    error:    str = ""


@dataclass
class VerificationReport:
    passed:  int
    total:   int
    results: list = field(default_factory=list)
    error:   str  = ""

    @property
    def reward(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def __str__(self):
        lines = [f"Passed: {self.passed}/{self.total}  |  Reward: {self.reward:.3f}"]
        if self.error:
            lines.append(f"Error: {self.error}")
        for r in self.results:
            icon = "✓" if r.passed else "✗"
            lines.append(f"  [{icon}] test {r.index + 1}" +
                         (f" — {r.error}" if r.error else ""))
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────

def run_io(code: str, stdin_str: str) -> tuple[str, str]:
    """Run code with stdin. Returns (stdout, error)."""
    old_stdin, old_stdout = sys.stdin, sys.stdout
    print("---")
    print(code)
    print(stdin_str)
    print("---")
    try:
        sys.stdin  = io.StringIO(stdin_str)
        sys.stdout = io.StringIO()
        exec(code, {})
        return sys.stdout.getvalue(), ""
    except Exception:
        return "", traceback.format_exc(limit=3)
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout


def run_fn(code: str, fn_name: str, args: list) -> tuple[any, str]:
    """Run code and call fn_name(*args). Returns (result, error)."""
    try:
        ns = {}
        exec(code, ns)
        return ns[fn_name](*args), ""
    except Exception:
        return None, traceback.format_exc(limit=3)


# ─────────────────────────────────────────────────────────────────────────────
# Normalise
# ─────────────────────────────────────────────────────────────────────────────

def normalise(s: str) -> str:
    return "\n".join(line.rstrip() for line in str(s).strip().splitlines())


# ─────────────────────────────────────────────────────────────────────────────
# Verifier
# ─────────────────────────────────────────────────────────────────────────────

def verify(
    solution:     str,
    input_output: dict,
    starter_code: str = "",
    max_tests:    int = None,
) -> VerificationReport:
    # Inject starter code if not already present
    if starter_code and starter_code.strip() not in solution:
        code = starter_code.strip() + "\n\n" + solution
    else:
        code = solution

    inputs   = input_output.get("inputs",  [])
    outputs  = input_output.get("outputs", [])
    fn_name  = input_output.get("fn_name")

    if max_tests:
        inputs  = inputs[:max_tests]
        outputs = outputs[:max_tests]

    results = []
    for i, (inp, expected) in enumerate(zip(inputs, outputs)):
        if fn_name:
            args = inp if isinstance(inp, list) else [inp]
            actual, err = run_fn(code, fn_name, args)
            if err:
                results.append(TestResult(i, False, error=err[:120]))
            else:
                passed = str(actual).strip() == str(expected).strip()
                results.append(TestResult(i, passed))
        else:
            actual, err = run_io(code, str(inp))
            if err:
                results.append(TestResult(i, False, error=err[:120]))
            else:
                passed = normalise(actual) == normalise(expected)
                results.append(TestResult(i, passed))

    passed = sum(r.passed for r in results)
    return VerificationReport(passed=passed, total=len(results), results=results)


def verify_sample(sample: dict, solution: str = None, max_tests: int = None) -> VerificationReport:
    """Pass a raw TACO sample. Uses first ground-truth solution if none provided."""
    raw_io = sample.get("test_cases") or "{}"

    print(sample.keys())
    print(raw_io)

    try:
        io_dict = json.loads(raw_io) if isinstance(raw_io, str) else raw_io
    except json.JSONDecodeError as e:
        return VerificationReport(0, 0, error=f"Bad input_output JSON: {e}")

    if not io_dict or not io_dict.get("inputs"):
        return VerificationReport(0, 0, error="No test cases available")

    if solution is None:
        solution = sample["answer"][0][10:-3]

    return verify(
        solution=solution,
        input_output=io_dict,
        starter_code=sample.get("starter_code") or "",
        max_tests=max_tests,
    )

# ── Load dataset ──────────────────────────────────────────────────────────────
ds = load_from_disk("/home/chanb/scratch/datasets/taco/data/taco_hint_sep/train")

for sample_i, sample in enumerate(ds):
    if sample_i > 1:
        break


    print("── Ground-truth solution ──")
    report = verify_sample(sample, max_tests=5)
    print(report)

    # print("\n── Broken solution ──")
    # report2 = verify_sample(sample, solution="print('wrong')", max_tests=5)
    # print(report2)
