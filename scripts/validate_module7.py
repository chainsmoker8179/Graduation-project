import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpha158_templates import build_templates


def main():
    templates = build_templates()
    total = sum(len(t["names"]) for t in templates)
    print("[templates] count:", len(templates))
    print("[templates] total factors:", total)

    # show a few templates
    for t in templates[:5]:
        print("\nTemplate", t["template_id"])
        print("names:", t["names"][:5], "...")
        print("params:", t["param_values"])
        print("expr examples:", t["expr_examples"])


if __name__ == "__main__":
    main()
