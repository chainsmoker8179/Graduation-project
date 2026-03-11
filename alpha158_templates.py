"""Build template computation graphs for Alpha158 expressions (module 7)."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Set


WINDOW_VALUES = {5, 10, 20, 30, 60}
Q_VALUES = {0.2, 0.8}

VAR_MAP = {
    "$open": "open_",
    "$high": "high_",
    "$low": "low_",
    "$close": "close_",
    "$volume": "volume_",
    "$vwap": "vwap_",
}


@dataclass
class Graph:
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    output: str = ""


class GraphBuilder(ast.NodeVisitor):
    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self._id = 0
        self.param_values: Dict[str, Set[float]] = {"N": set(), "Q": set()}

    def _new_id(self) -> str:
        nid = f"n{self._id}"
        self._id += 1
        return nid

    def _add_node(self, op: str, inputs=None, **kwargs) -> str:
        nid = self._new_id()
        node = {"id": nid, "op": op}
        if inputs:
            node["inputs"] = inputs
        for k, v in kwargs.items():
            node[k] = v
        self.nodes.append(node)
        return nid

    def visit_Name(self, node: ast.Name):
        return self._add_node("var", name=node.id)

    def visit_Constant(self, node: ast.Constant):
        val = node.value
        if isinstance(val, (int, float)):
            # Window placeholder
            if isinstance(val, int) and val in WINDOW_VALUES:
                self.param_values["N"].add(float(val))
                return self._add_node("param", name="N")
            # Quantile placeholder
            if isinstance(val, float) and any(abs(val - q) < 1e-9 for q in Q_VALUES):
                self.param_values["Q"].add(float(val))
                return self._add_node("param", name="Q")
        return self._add_node("const", value=val)

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            op = "Add"
        elif isinstance(node.op, ast.Sub):
            op = "Sub"
        elif isinstance(node.op, ast.Mult):
            op = "Mul"
        elif isinstance(node.op, ast.Div):
            op = "Div"
        elif isinstance(node.op, ast.Pow):
            op = "Pow"
        else:
            raise ValueError(f"Unsupported binop: {node.op}")
        return self._add_node(op, [left, right])

    def visit_UnaryOp(self, node: ast.UnaryOp):
        val = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            c = self._add_node("const", value=-1.0)
            return self._add_node("Mul", [c, val])
        if isinstance(node.op, ast.UAdd):
            return val
        raise ValueError(f"Unsupported unary op: {node.op}")

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            op = node.func.id
        else:
            raise ValueError("Unsupported function call")
        inputs = [self.visit(a) for a in node.args]
        return self._add_node(op, inputs)

    def visit_Compare(self, node: ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Only simple comparisons supported")
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        if isinstance(node.ops[0], ast.Gt):
            op = "Gt"
        elif isinstance(node.ops[0], ast.Lt):
            op = "Lt"
        elif isinstance(node.ops[0], ast.GtE):
            op = "Ge"
        elif isinstance(node.ops[0], ast.LtE):
            op = "Le"
        else:
            raise ValueError(f"Unsupported compare op: {node.ops[0]}")
        return self._add_node(op, [left, right])


def _preprocess_expr(expr: str) -> str:
    # Replace $vars with python identifiers
    for k, v in VAR_MAP.items():
        expr = expr.replace(k, v)
    return expr


def _graph_signature(nodes: List[Dict[str, Any]]) -> Tuple:
    id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
    sig = []
    for n in nodes:
        op = n["op"]
        inputs = tuple(id_to_idx[i] for i in n.get("inputs", []))
        if op == "var":
            sig.append((op, n["name"]))
        elif op == "const":
            sig.append((op, n["value"]))
        elif op == "param":
            sig.append((op, n["name"]))
        else:
            sig.append((op, inputs))
    return tuple(sig)


def build_templates(csv_path: str = "alpha158_name_expression.csv") -> List[Dict[str, Any]]:
    templates: Dict[Tuple, Dict[str, Any]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            expr = row["expression"]

            expr_py = _preprocess_expr(expr)
            tree = ast.parse(expr_py, mode="eval")
            builder = GraphBuilder()
            out_id = builder.visit(tree.body)
            graph = Graph(nodes=builder.nodes, output=out_id)

            sig = _graph_signature(graph.nodes)
            if sig not in templates:
                templates[sig] = {
                    "template_id": f"T{len(templates)+1}",
                    "graph": graph,
                    "param_values": {"N": set(), "Q": set()},
                    "names": [],
                    "name_params": {},
                    "expr_examples": [],
                }
            t = templates[sig]
            t["names"].append(name)
            if len(t["expr_examples"]) < 3:
                t["expr_examples"].append(expr)
            # merge params
            for k, vset in builder.param_values.items():
                t["param_values"][k].update(vset)
            # record params for this specific factor
            params = {}
            for k, vset in builder.param_values.items():
                if vset:
                    if len(vset) > 1:
                        raise ValueError(f"Expression {name} has multiple {k} values: {vset}")
                    params[k] = list(vset)[0]
            t["name_params"][name] = params

    # convert sets to sorted lists
    out = []
    for t in templates.values():
        t["param_values"] = {k: sorted(v) for k, v in t["param_values"].items() if v}
        out.append(t)
    return out


def graph_to_dot(graph: Graph) -> str:
    """Convert a Graph to Graphviz DOT format (for visualization)."""
    lines = ["digraph G {", "  rankdir=LR;"]
    # Nodes
    for n in graph.nodes:
        nid = n["id"]
        op = n["op"]
        if op == "var":
            name = n["name"]
            if name.endswith("_"):
                name = name[:-1]
            label = f"var:{name}"
        elif op == "const":
            label = f"const:{n['value']}"
        elif op == "param":
            label = f"param:{n['name']}"
        else:
            label = op
        label = str(label).replace("\"", "\\\"")
        lines.append(f'  {nid} [label="{label}"];')
    # Edges
    for n in graph.nodes:
        for inp in n.get("inputs", []):
            lines.append(f"  {inp} -> {n['id']};")
    lines.append("}")
    return "\n".join(lines)


__all__ = ["build_templates", "graph_to_dot", "WINDOW_VALUES", "Q_VALUES"]
