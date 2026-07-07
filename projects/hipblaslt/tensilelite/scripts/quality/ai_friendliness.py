"""AI-friendliness AST scanner for the tensilelite Tensile/ Python tree.

Stdlib-only. Walks the Python source tree and produces a set of readability /
"AI-friendliness" signals: counts of files / functions / call sites that trip
heuristics known to make a codebase harder for both humans and LLM agents to
read, navigate, and safely change.

Ported and adapted from the rocMETRICS quality-gate scanner for the Tensile/
layout:

  * tests live under ``Tensile/Tests`` (nested, not a sibling ``tests/`` dir),
  * ``Common`` and ``Utilities`` are treated as shared infrastructure,
  * the "adapter" allowlist is empty (no adapter layer in this tree).

The scanner is intentionally conservative: every signal is a *count*, never a
hard failure, so the caller can render a report-only "current vs target" table.
"""

from __future__ import annotations

import ast
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# --------------------------------------------------------------------------- #
# Tunables (thresholds only decide what *counts*, never whether we fail).
# --------------------------------------------------------------------------- #

MAX_FILE_LOC = 1000          # files >= this are "very large"
LARGE_FILE_LOC = 509         # files > this are "large" (mirrors NLOC target)
MAX_NESTING_DEPTH = 4        # nesting deeper than this is "deep"
MAX_FUNCTION_ARGS = 6        # more positional/keyword params than this is "wide"
MAX_RETURNS = 6              # more return statements than this is "many-exit"
MIN_DUP_LITERAL = 5          # a string literal repeated >= this many times
BROAD_EXCEPTIONS = {"Exception", "BaseException"}

# Directory roles within the Tensile/ tree.
TEST_DIR_PARTS = ("Tests",)
SHARED_INFRA_DIRS = ("Common", "Utilities")
ADAPTER_ALLOWLIST: Set[str] = set()  # empty for Tensile/


@dataclass
class FileFacts:
    """Per-file measurements collected in a single AST pass."""

    path: Path
    loc: int = 0
    is_test: bool = False
    is_shared_infra: bool = False
    functions: int = 0
    classes: int = 0
    max_nesting: int = 0
    deep_functions: int = 0
    wide_signatures: int = 0
    many_exit_functions: int = 0
    swallowed_errors: int = 0
    broad_excepts: int = 0
    bare_excepts: int = 0
    todo_comments: int = 0
    print_calls: int = 0
    star_imports: int = 0
    typing_any: int = 0
    type_ignores: int = 0
    global_statements: int = 0
    mutable_defaults: int = 0
    long_lines: int = 0
    module_name: str = ""
    imported_modules: Set[str] = field(default_factory=set)
    string_literals: Counter = field(default_factory=Counter)
    has_docstring: bool = False
    parse_error: bool = False


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #


def _is_test_path(rel_parts: Tuple[str, ...]) -> bool:
    return any(part in TEST_DIR_PARTS for part in rel_parts)


def _is_shared_infra_path(rel_parts: Tuple[str, ...]) -> bool:
    return any(part in SHARED_INFRA_DIRS for part in rel_parts)


def _feature_of(rel_parts: Tuple[str, ...]) -> str:
    """First path component under the tree root; used for cross-feature imports."""
    return rel_parts[0] if rel_parts else ""


def _node_nesting(node: ast.AST, depth: int = 0) -> int:
    """Max control-flow nesting depth under ``node``."""
    nest_types = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith)
    best = depth
    for child in ast.iter_child_nodes(node):
        child_depth = depth + 1 if isinstance(child, nest_types) else depth
        best = max(best, _node_nesting(child, child_depth))
    return best


def _except_swallows(handler: ast.ExceptHandler) -> bool:
    """A handler that neither re-raises, logs, nor does anything but pass/continue."""
    meaningful = False
    reraises = False
    for stmt in ast.walk(handler):
        if isinstance(stmt, ast.Raise):
            reraises = True
        if isinstance(stmt, ast.Call):
            meaningful = True
    if reraises:
        return False
    body = handler.body
    if len(body) == 1 and isinstance(body[0], (ast.Pass,)):
        return True
    if not meaningful and all(
        isinstance(s, (ast.Pass, ast.Continue, ast.Break)) for s in body
    ):
        return True
    return False


def _has_mutable_default(func: ast.AST) -> bool:
    args = getattr(func, "args", None)
    if args is None:
        return False
    defaults = list(getattr(args, "defaults", [])) + [
        d for d in getattr(args, "kw_defaults", []) if d is not None
    ]
    return any(isinstance(d, (ast.List, ast.Dict, ast.Set)) for d in defaults)


def _signature_width(func: ast.AST) -> int:
    args = getattr(func, "args", None)
    if args is None:
        return 0
    count = len(args.args) + len(getattr(args, "posonlyargs", [])) + len(args.kwonlyargs)
    if args.vararg:
        count += 1
    if args.kwarg:
        count += 1
    return count


# --------------------------------------------------------------------------- #
# Per-file scan
# --------------------------------------------------------------------------- #


def scan_file(path: Path, root: Path) -> FileFacts:
    rel = path.relative_to(root)
    rel_parts = rel.parts
    facts = FileFacts(path=path)
    facts.is_test = _is_test_path(rel_parts)
    facts.is_shared_infra = _is_shared_infra_path(rel_parts)
    facts.module_name = path.stem

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        facts.parse_error = True
        return facts

    lines = text.splitlines()
    facts.loc = len(lines)
    for line in lines:
        stripped = line.strip()
        if len(line) > 100:
            facts.long_lines += 1
        if "# type: ignore" in line or "# type:ignore" in line:
            facts.type_ignores += 1
        upper = stripped.upper()
        if upper.startswith("#") and ("TODO" in upper or "FIXME" in upper or "XXX" in upper):
            facts.todo_comments += 1

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        facts.parse_error = True
        return facts

    facts.has_docstring = ast.get_docstring(tree) is not None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            facts.functions += 1
            nesting = _node_nesting(node)
            facts.max_nesting = max(facts.max_nesting, nesting)
            if nesting > MAX_NESTING_DEPTH:
                facts.deep_functions += 1
            if _signature_width(node) > MAX_FUNCTION_ARGS:
                facts.wide_signatures += 1
            returns = sum(1 for n in ast.walk(node) if isinstance(n, ast.Return))
            if returns > MAX_RETURNS:
                facts.many_exit_functions += 1
            if _has_mutable_default(node):
                facts.mutable_defaults += 1
        elif isinstance(node, ast.ClassDef):
            facts.classes += 1
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                facts.bare_excepts += 1
            else:
                names = {
                    n.id for n in ast.walk(node.type) if isinstance(n, ast.Name)
                }
                if names & BROAD_EXCEPTIONS:
                    facts.broad_excepts += 1
            if _except_swallows(node):
                facts.swallowed_errors += 1
        elif isinstance(node, ast.Global):
            facts.global_statements += 1
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                facts.star_imports += 1
            if node.module:
                facts.imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                facts.imported_modules.add(alias.name)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                facts.print_calls += 1
        elif isinstance(node, ast.Attribute):
            if node.attr == "Any":
                facts.typing_any += 1
        elif isinstance(node, ast.Name):
            if node.id == "Any":
                facts.typing_any += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if 3 <= len(val) <= 60 and val.strip():
                facts.string_literals[val] += 1

    return facts


# --------------------------------------------------------------------------- #
# Tree aggregation -> 21 AI-friendliness signals
# --------------------------------------------------------------------------- #


def iter_python_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in {"__pycache__", ".git", "build", "dist", ".tox", ".venv"}
        ]
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def scan_tree(root: Path) -> Dict[str, int]:
    """Return an ordered dict of the 21 AI-friendliness signals for ``root``."""
    root = root.resolve()
    facts_list: List[FileFacts] = [scan_file(p, root) for p in iter_python_files(root)]

    total_files = len(facts_list)
    src_facts = [f for f in facts_list if not f.is_test]
    test_facts = [f for f in facts_list if f.is_test]

    # Cross-feature imports: a source file importing a sibling top-level feature.
    features_by_file: Dict[Path, str] = {}
    for f in facts_list:
        rel = f.path.relative_to(root)
        features_by_file[f.path] = _feature_of(rel.parts)
    feature_names = {v for v in features_by_file.values() if v and not v.endswith(".py")}

    cross_feature_imports = 0
    for f in src_facts:
        own = features_by_file[f.path]
        for mod in f.imported_modules:
            head = mod.split(".")[0]
            if head in feature_names and head != own:
                cross_feature_imports += 1
                break

    # Duplicated string literals across the tree.
    global_literals: Counter = Counter()
    for f in facts_list:
        global_literals.update(f.string_literals)
    duplicated_literals = sum(1 for _, c in global_literals.items() if c >= MIN_DUP_LITERAL)

    # Missing seam tests: source modules with no same-named test module.
    test_module_names = {f.module_name.replace("test_", "").replace("_test", "")
                         for f in test_facts}
    test_module_names |= {f.module_name for f in test_facts}
    missing_seam_tests = sum(
        1 for f in src_facts
        if not f.is_shared_infra and f.module_name not in test_module_names
        and f.module_name != "__init__"
    )

    signals: Dict[str, int] = {
        "total_python_files": total_files,
        "source_files": len(src_facts),
        "test_files": len(test_facts),
        "very_large_files": sum(1 for f in facts_list if f.loc >= MAX_FILE_LOC),
        "large_files": sum(1 for f in facts_list if f.loc > LARGE_FILE_LOC),
        "files_without_module_docstring": sum(
            1 for f in src_facts if not f.has_docstring and f.module_name != "__init__"
        ),
        "deeply_nested_functions": sum(f.deep_functions for f in facts_list),
        "wide_signatures": sum(f.wide_signatures for f in facts_list),
        "many_exit_functions": sum(f.many_exit_functions for f in facts_list),
        "swallowed_errors": sum(f.swallowed_errors for f in facts_list),
        "broad_except_clauses": sum(f.broad_excepts for f in facts_list),
        "bare_except_clauses": sum(f.bare_excepts for f in facts_list),
        "mutable_default_args": sum(f.mutable_defaults for f in facts_list),
        "global_statements": sum(f.global_statements for f in facts_list),
        "star_imports": sum(f.star_imports for f in facts_list),
        "typing_any_uses": sum(f.typing_any for f in facts_list),
        "type_ignore_comments": sum(f.type_ignores for f in facts_list),
        "print_call_sites": sum(f.print_calls for f in src_facts),
        "todo_fixme_comments": sum(f.todo_comments for f in facts_list),
        "cross_feature_imports": cross_feature_imports,
        "duplicated_literals": duplicated_literals,
        "missing_seam_tests": missing_seam_tests,
        "long_lines": sum(f.long_lines for f in facts_list),
    }
    return signals
