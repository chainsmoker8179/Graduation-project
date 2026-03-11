import os
import sys
import argparse
import shutil
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpha158_templates import build_templates, graph_to_dot


def main():
    parser = argparse.ArgumentParser(description="Render Alpha158 template graph to DOT/PNG")
    parser.add_argument("--template", type=str, default="T1", help="Template id, e.g., T1")
    parser.add_argument("--out", type=str, default="", help="Output basename without extension")
    parser.add_argument("--out-dir", type=str, default="graph_images", help="Output directory for images")
    parser.add_argument("--all", action="store_true", help="Render all templates")
    parser.add_argument("--count", type=int, default=0, help="Render first N templates (0 means disabled)")
    args = parser.parse_args()

    templates = build_templates()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        selected = templates
    elif args.count and args.count > 0:
        selected = templates[: args.count]
    else:
        selected = [next((t for t in templates if t["template_id"] == args.template), None)]
        if selected[0] is None:
            raise SystemExit(f"Template {args.template} not found")

    dot_bin = shutil.which("dot")
    if not dot_bin:
        print("Graphviz 'dot' not found; PNG not generated.")

    for t in selected:
        template_id = t["template_id"]
        out_base = args.out or f"{template_id}_graph"
        out_base = out_dir / out_base
        dot_path = str(out_base) + ".dot"
        dot_str = graph_to_dot(t["graph"])
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write(dot_str)
        print(f"Wrote DOT: {dot_path}")

        if dot_bin:
            png_path = str(out_base) + ".png"
            os.system(f"{dot_bin} -Tpng {dot_path} -o {png_path}")
            print(f"Wrote PNG: {png_path}")


if __name__ == "__main__":
    main()
