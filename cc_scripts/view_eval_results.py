#!/usr/bin/env python3
"""
Quick reference: View evaluation results summary
Usage: python view_eval_results.py
"""

import json
from pathlib import Path


def print_table(title, data):
    """Pretty-print results as aligned table"""
    print(f"\n{title}")
    print("=" * 90)

    if isinstance(data, dict) and len(data) > 0:
        # Get all keys
        keys = (
            list(list(data.values())[0].keys())
            if isinstance(list(data.values())[0], dict)
            else []
        )

        # Header
        col_width = 18
        header = f"{'Dataset':<{col_width}}"
        for key in keys:
            header += f"{key.upper():<{col_width}}"
        print(header)
        print("-" * len(header))

        # Rows
        for dataset, values in data.items():
            if isinstance(values, dict):
                row = f"{dataset.upper():<{col_width}}"
                for key in keys:
                    val = values.get(key, 0)
                    row += (
                        f"{val:.1%}"
                        if isinstance(val, (int, float))
                        else f"{str(val):<{col_width}}"
                    )
                print(row)


def main():
    results_file = Path("evaluation_results.json")

    if not results_file.exists():
        print("❌ evaluation_results.json not found. Run: python evaluation_results.py")
        return

    with open(results_file) as f:
        data = json.load(f)

    # Header
    print("\n" + "=" * 90)
    print("HINTRL EVALUATION RESULTS — COMPLETE SUMMARY")
    print("=" * 90)
    print(f"Date: {data['metadata']['date']}")
    print(f"Total samples: {data['metadata']['total_samples']:,}")
    print(f"Completion: {data['metadata']['completion']}")
    print(f"Data integrity: {data['metadata']['data_integrity']}")

    # Pass@1
    print_table("\nPass@1 (Single Sample Accuracy)", data["results"]["pass_at_1"])

    # Pass@8
    print_table("\nPass@8 (Best of 8 Samples)", data["results"]["pass_at_8"])

    # Pass@32
    print_table("\nPass@32 (Best of 32 Samples)", data["results"]["pass_at_32"])

    # Key findings
    print("\n" + "=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)

    for key, finding in data["key_findings"].items():
        print(f"\n{key.upper().replace('_', ' ')}:")
        for k, v in finding.items():
            if not isinstance(v, dict):
                print(f"  • {k.replace('_', ' ').title()}: {v}")

    # Recommendations
    print("\n" + "=" * 90)
    print("RECOMMENDATIONS")
    print("=" * 90)

    for phase, items in data["recommendations"].items():
        print(f"\n{phase.upper().replace('_', ' ')}:")
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item}")

    print("\n" + "=" * 90)
    print("Full details: cat cc_scripts/EVALUATION_COMPLETE.md")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
