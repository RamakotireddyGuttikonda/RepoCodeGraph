import ast
import builtins
import colorsys
import inspect
import json
import os
import pickle
import random
import re
import sys
import warnings
from collections import defaultdict, namedtuple
from copy import deepcopy
from pathlib import Path, PurePosixPath

import networkx as nx

warnings.simplefilter("ignore", category=FutureWarning)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # noqa: E302
        return iterable

from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tree_sitter_languages import get_language, get_parser

Tag = namedtuple("Tag", "rel_fname fname line name kind category info".split())


# ---------------------------------------------------------------------------
# Bundled create_structure  (replaces `from utils import create_structure`)
# ---------------------------------------------------------------------------

def create_structure(root: str) -> dict:
    """
    Recursively walk *root* and build a nested dict that mirrors the
    directory tree.  Each Python file leaf contains:

        {
            "classes":   [ { "name", "start_line", "end_line", "methods": [...] } ],
            "functions": [ { "name", "start_line", "end_line", "text": [...] } ],
        }

    Non-Python files / directories that cannot be parsed are silently skipped.
    """

    def _parse_file(path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
            tree = ast.parse(source)
        except Exception:
            return {"classes": [], "functions": []}

        lines = source.splitlines()

        classes = []
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_start = item.lineno - 1
                        m_end   = item.end_lineno - 1 if hasattr(item, "end_lineno") else m_start
                        methods.append({
                            "name":       item.name,
                            "start_line": m_start,
                            "end_line":   m_end,
                            "text":       lines[m_start: m_end + 1],
                        })
                c_start = node.lineno - 1
                c_end   = node.end_lineno - 1 if hasattr(node, "end_lineno") else c_start
                classes.append({
                    "name":       node.name,
                    "start_line": c_start,
                    "end_line":   c_end,
                    "methods":    methods,
                })

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # top-level functions only (skip methods captured above)
                if not isinstance(getattr(node, "_parent", None), ast.ClassDef):
                    f_start = node.lineno - 1
                    f_end   = node.end_lineno - 1 if hasattr(node, "end_lineno") else f_start
                    functions.append({
                        "name":       node.name,
                        "start_line": f_start,
                        "end_line":   f_end,
                        "text":       lines[f_start: f_end + 1],
                    })

        # Mark parents so we can skip methods above (second pass not needed with the
        # approach below, but kept for clarity)
        return {"classes": classes, "functions": functions}

    def _build(directory: str) -> dict:
        result: dict = {}
        try:
            entries = sorted(os.listdir(directory))
        except PermissionError:
            return result

        for entry in entries:
            full = os.path.join(directory, entry)
            if os.path.isdir(full):
                subtree = _build(full)
                if subtree:          # only add non-empty sub-dicts
                    result[entry] = subtree
            elif entry.endswith(".py"):
                result[entry] = _parse_file(full)

        return result

    return _build(root)


# ---------------------------------------------------------------------------
# CodeGraph
# ---------------------------------------------------------------------------

class CodeGraph:
    warned_files: set = set()

    def __init__(
        self,
        map_tokens: int = 1024,
        root: str | None = None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose: bool = False,
        max_context_window=None,
    ):
        self.io = io or _SimpleIO()
        self.verbose = verbose
        self.root = root or os.getcwd()
        self.max_map_tokens = map_tokens
        self.max_context_window = max_context_window
        self.repo_content_prefix = repo_content_prefix
        self.tree_cache: dict = {}
        self.structure = create_structure(self.root)

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

        # ------------------- Add this method inside CodeGraph class -------------------
    def draw_graph(self, G, output_path="code_graph.png", max_nodes=150):
        """Generate and save a visual representation of the code graph."""
        import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        if len(G.nodes) == 0:
            print("Graph is empty, skipping visualization.")
            return False

        if len(G.nodes) > max_nodes:
            print(f"Graph too large ({len(G.nodes)} nodes). Drawing only top {max_nodes} nodes.")
            # Take largest connected component or most central nodes
            if nx.is_connected(G.to_undirected()):
                nodes = list(G.nodes)[:max_nodes]
            else:
                largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
                nodes = list(largest_cc)[:max_nodes]
            G = G.subgraph(nodes).copy()

        plt.figure(figsize=(14, 10), dpi=300)

        # Layout
        try:
            pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        except:
            pos = nx.random_layout(G)

        # Node colors by category
        color_map = []
        for node in G.nodes:
            cat = G.nodes[node].get('category', 'function')
            if cat == 'class':
                color_map.append('#e74c3c')      # red
            else:
                color_map.append('#3498db')      # blue

        # Draw
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=color_map,
            node_size=800,
            font_size=7,
            font_color='white',
            font_weight='bold',
            edge_color='gray',
            alpha=0.85,
            arrows=True,
            arrowsize=15,
            width=1.2
        )

        plt.title(f"Code Graph Visualization\n{len(G.nodes)} Nodes • {len(G.edges)} Edges", 
                  fontsize=14, pad=20)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', label='Class'),
            Patch(facecolor='#3498db', label='Function')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Graph image saved → {output_path}")
        return True

    def get_code_graph(self, other_files, mentioned_fnames=None):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()

        tags = self.get_tag_files(other_files, mentioned_fnames)
        code_graph = self.tag_to_graph(tags)
        return tags, code_graph

    # ------------------------------------------------------------------
    # Tag collection
    # ------------------------------------------------------------------

    def get_tag_files(self, other_files, mentioned_fnames=None):
        try:
            return self.get_ranked_tags(other_files, mentioned_fnames)
        except RecursionError:
            self.io.tool_error("Disabling code graph, git repo too large?")
            self.max_map_tokens = 0
            return []

    def get_ranked_tags(self, other_fnames, mentioned_fnames):
        tags_of_files: list = []
        personalization: dict = {}
        fnames = sorted(set(other_fnames))
        personalize = 10 / max(len(fnames), 1)

        for fname in tqdm(fnames, desc="Parsing files"):
            fpath = Path(fname)
            if not fpath.is_file():
                if fname not in self.warned_files:
                    if fpath.exists():
                        self.io.tool_error(f"Code graph can't include {fname}, not a normal file")
                    else:
                        self.io.tool_error(f"Code graph can't include {fname}, it no longer exists")
                    self.warned_files.add(fname)
                continue

            rel_fname = self.get_rel_fname(fname)
            if fname in mentioned_fnames:
                personalization[rel_fname] = personalize

            tags = list(self.get_tags(fname, rel_fname))
            tags_of_files.extend(tags)

        return tags_of_files

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def tag_to_graph(self, tags):
        G = nx.MultiDiGraph()
        for tag in tags:
            G.add_node(
                tag["name"],
                category=tag["category"],
                info=tag["info"],
                fname=tag["fname"],
                line=tag["line"],
                kind=tag["kind"],
            )

        for tag in tags:
            if tag["category"] == "class":
                for f in tag["info"].split("\t"):
                    f = f.strip()
                    if f:
                        G.add_edge(tag["name"], f)

        tags_ref = [t for t in tags if t["kind"] == "ref"]
        tags_def = [t for t in tags if t["kind"] == "def"]
        def_names = {t["name"] for t in tags_def}
        for tag in tags_ref:
            if tag["name"] in def_names:
                G.add_edge(tag["name"], tag["name"])

        return G

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_rel_fname(self, fname: str) -> str:
        return os.path.relpath(fname, self.root)

    def split_path(self, path: str) -> list:
        return [os.path.relpath(path, self.root) + ":"]

    def get_mtime(self, fname: str):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.io.tool_error(f"File not found: {fname}")
            return None

    # ------------------------------------------------------------------
    # Structure navigation  (Windows-safe: split on os.sep or '/')
    # ------------------------------------------------------------------

    def _navigate_structure(self, rel_fname: str) -> dict:
        """Walk self.structure using the parts of rel_fname."""
        # Normalise to forward-slash parts so it works on both platforms
        parts = Path(rel_fname).parts   # e.g. ('subpkg', 'module.py')
        s = deepcopy(self.structure)
        for part in parts:
            if part not in s:
                raise KeyError(f"Part '{part}' not found in structure (rel_fname={rel_fname!r})")
            s = s[part]
        return s

    # ------------------------------------------------------------------
    # AST helpers
    # ------------------------------------------------------------------

    def get_class_functions(self, tree, class_name: str) -> list:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return [item.name for item in node.body
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return []

    def get_func_block(self, first_line: str, code_block: str):
        first_line_escaped = re.escape(first_line)
        pattern = re.compile(
            rf"({first_line_escaped}.*?)(?=(^\S|\Z))", re.DOTALL | re.MULTILINE
        )
        match = pattern.search(code_block)
        return match.group(0) if match else None

    # ------------------------------------------------------------------
    # Standard-library / third-party function detection
    # ------------------------------------------------------------------

    def std_proj_funcs(self, code: str, fname: str):
        """Return (std_funcs, std_libs) imported by *code* that are not project-local."""
        std_libs: list = []
        std_funcs: list = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return std_funcs, std_libs

        codelines = code.split("\n")
        fname_norm = fname.replace("\\", "/")   # normalise for comparison

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_stmt = codelines[node.lineno - 1].strip()
                for alias in node.names:
                    import_name = alias.name.split(".")[0]
                    if import_name in fname_norm:
                        continue
                    local_ns: dict = {}
                    try:
                        exec(import_stmt, {}, local_ns)   # noqa: S102
                    except Exception:
                        continue
                    std_libs.append(alias.name)
                    eval_name = alias.asname if alias.asname else alias.name
                    obj = local_ns.get(eval_name)
                    if obj is not None:
                        std_funcs.extend(
                            name for name, member in inspect.getmembers(obj)
                            if callable(member)
                        )

            elif isinstance(node, ast.ImportFrom):
                import_stmt = codelines[node.lineno - 1]
                if node.module is None:
                    continue
                module_name = node.module.split(".")[0]
                if module_name in fname_norm:
                    continue

                # handle multi-line imports with parentheses
                if "(" in import_stmt:
                    end_ln = node.lineno - 1
                    for ln in range(node.lineno - 1, len(codelines)):
                        if ")" in codelines[ln]:
                            end_ln = ln
                            break
                    import_stmt = "\n".join(codelines[node.lineno - 1: end_ln + 1])

                import_stmt = import_stmt.strip()
                local_ns: dict = {}
                try:
                    exec(import_stmt, {}, local_ns)   # noqa: S102
                except Exception:
                    continue

                for alias in node.names:
                    std_libs.append(alias.name)
                    eval_name = alias.asname if alias.asname else alias.name
                    if eval_name == "*":
                        continue
                    obj = local_ns.get(eval_name)
                    if obj is not None:
                        std_funcs.extend(
                            name for name, member in inspect.getmembers(obj)
                            if callable(member)
                        )

        return std_funcs, std_libs

    # ------------------------------------------------------------------
    # Core tag extraction
    # ------------------------------------------------------------------

    def get_tags(self, fname: str, rel_fname: str) -> list:
        if self.get_mtime(fname) is None:
            return []
        return list(self.get_tags_raw(fname, rel_fname))

    def get_tags_raw(self, fname: str, rel_fname: str):  # noqa: C901  (complex but intentional)
        # ---- navigate structure dict ----
        try:
            s = self._navigate_structure(rel_fname)
        except (KeyError, TypeError):
            return

        structure_classes      = {item["name"]: item for item in s.get("classes", [])}
        structure_functions    = {item["name"]: item for item in s.get("functions", [])}
        structure_class_methods: dict = {}
        for cls in s.get("classes", []):
            for item in cls.get("methods", []):
                structure_class_methods[item["name"]] = item
        structure_all_funcs = {**structure_functions, **structure_class_methods}

        lang = filename_to_lang(fname)
        if not lang:
            return
        language = get_language(lang)
        parser   = get_parser(lang)

        query_scm = """
(class_definition
  name: (identifier) @name.definition.class) @definition.class

(function_definition
  name: (identifier) @name.definition.function) @definition.function

(call
  function: [
    (identifier) @name.reference.call
    (attribute
      attribute: (identifier) @name.reference.call)
  ]) @reference.call
"""

        try:
            with open(fname, "r", encoding="utf-8", errors="replace") as fh:
                code      = fh.read()
            with open(fname, "r", encoding="utf-8", errors="replace") as fh:
                codelines = fh.readlines()
        except OSError:
            return

        # Sanitise known edge-cases that trip the tree-sitter parser
        code = code.replace("\ufeff", "")
        code = code.replace("constants.False", "_False")
        code = code.replace("constants.True",  "_True")
        code = code.replace("False", "_False")
        code = code.replace("True",  "_True")
        code = code.replace("DOMAIN\\username", "DOMAIN\\\\username")
        code = code.replace("Error, ",     "Error as ")
        code = code.replace("Exception, ", "Exception as ")
        code = code.replace("print ", "yield ")
        code = re.sub(r"except\s+\(([^,]+)\s+as\s+([^)]+)\):", r"except (\1, \2):", code)
        code = code.replace("raise AttributeError as aname", "raise AttributeError")

        if not code:
            return

        tree = parser.parse(bytes(code, "utf-8"))

        try:
            std_funcs, std_libs = self.std_proj_funcs(code, fname)
        except Exception:
            std_funcs, std_libs = [], []

        builtin_names = (
            [name for name in dir(builtins)]
            + dir(list) + dir(dict) + dir(set)
            + dir(str)  + dir(tuple)
        )

        query    = language.query(query_scm)
        captures = list(query.captures(tree.root_node))

        saw: set = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)
            cur_cdl   = codelines[node.start_point[0]]
            category  = "class" if "class " in cur_cdl else "function"
            tag_name  = node.text.decode("utf-8")

            if tag_name in std_funcs or tag_name in std_libs or tag_name in builtin_names:
                continue

            if category == "class":
                if kind == "def":
                    cls_info = structure_classes.get(tag_name)
                    if cls_info is None:
                        continue
                    class_functions = [m["name"] for m in cls_info.get("methods", [])]
                    line_nums = [cls_info["start_line"], cls_info["end_line"]]
                else:
                    class_functions = []
                    line_nums = [node.start_point[0], node.end_point[0]]

                yield {
                    "rel_fname": rel_fname,
                    "fname":     fname,
                    "name":      tag_name,
                    "kind":      kind,
                    "category":  category,
                    "info":      "\n".join(class_functions),
                    "line":      line_nums,
                }

            else:  # function
                if kind == "def":
                    func_info = structure_all_funcs.get(tag_name)
                    if func_info is None:
                        continue
                    cur_cdl  = "\n".join(func_info.get("text", []))
                    line_nums = [func_info["start_line"], func_info["end_line"]]
                else:
                    line_nums = [node.start_point[0], node.end_point[0]]

                yield {
                    "rel_fname": rel_fname,
                    "fname":     fname,
                    "name":      tag_name,
                    "kind":      kind,
                    "category":  category,
                    "info":      cur_cdl,
                    "line":      line_nums,
                }

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # Fallback: use Pygments tokens as refs when tree-sitter only found defs
        try:
            lexer  = guess_lexer_for_filename(fname, code)
            tokens = [tok[1] for tok in lexer.get_tokens(code) if tok[0] in Token.Name]
        except ClassNotFound:
            return

        for token in tokens:
            yield {
                "rel_fname": rel_fname,
                "fname":     fname,
                "name":      token,
                "kind":      "ref",
                "category":  "function",
                "info":      "none",
                "line":      [-1, -1],
            }

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def find_src_files(self, directory: str) -> list:
        if not os.path.isdir(directory):
            return [directory]
        src_files = []
        for root, _dirs, files in os.walk(directory):
            for f in files:
                src_files.append(os.path.join(root, f))
        return src_files

    def find_files(self, dirs) -> list:
        chat_fnames = []
        for fname in dirs:
            if Path(fname).is_dir():
                chat_fnames.extend(self.find_src_files(fname))
            else:
                chat_fnames.append(fname)
        return [f for f in chat_fnames if f.endswith(".py")]

    # ------------------------------------------------------------------
    # Tree rendering (kept for compatibility; requires tree_cache attr)
    # ------------------------------------------------------------------

    def render_tree(self, abs_fname: str, rel_fname: str, lois: list) -> str:
        key = (rel_fname, tuple(sorted(lois)))
        if key in self.tree_cache:
            return self.tree_cache[key]
        with open(abs_fname, "r", encoding="utf-8", errors="replace") as fh:
            code = fh.read() or ""
        if not code.endswith("\n"):
            code += "\n"
        context = TreeContext(
            rel_fname, code,
            color=False, line_number=False, child_context=False,
            last_line=False, margin=0, mark_lois=False, loi_pad=0,
            show_top_of_file_parent_scope=False,
        )
        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res


# ---------------------------------------------------------------------------
# Minimal IO shim (used when no io object is injected)
# ---------------------------------------------------------------------------

class _SimpleIO:
    @staticmethod
    def tool_error(msg: str):
        print(f"[ERROR] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_random_color() -> str:
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python construct_graph.py <dir_to_repo>")
        sys.exit(1)

    dir_name = sys.argv[1]
    print(f"Analysing: {dir_name}")

    code_graph = CodeGraph(root=dir_name)
    chat_fnames = code_graph.find_files([dir_name])

    print(f"Found {len(chat_fnames)} Python file(s)")

    if not chat_fnames:
        print("No Python files found.")
        sys.exit(0)

    result = code_graph.get_code_graph(chat_fnames)
    if result is None or result[1] is None:
        print("Failed to build graph.")
        sys.exit(1)

    tags, G = result

    print("-" * 60)
    print(f"✅ Successfully built the code graph for: {dir_name}")
    print(f" Nodes : {len(G.nodes)}")
    print(f" Edges : {len(G.edges)}")
    print("-" * 60)

    out_dir = os.getcwd()

    # Save graph & tags
    with open(os.path.join(out_dir, "graph.pkl"), "wb") as fh:
        pickle.dump(G, fh)

    with open(os.path.join(out_dir, "tags.jsonl"), "w", encoding="utf-8") as fh:
        for tag in tags:
            line = json.dumps({
                "fname": tag["fname"],
                "rel_fname": tag["rel_fname"],
                "line": tag["line"],
                "name": tag["name"],
                "kind": tag["kind"],
                "category": tag["category"],
                "info": tag["info"],
            }, ensure_ascii=False)
            fh.write(line + "\n")

    # === Generate Image ===
    code_graph.draw_graph(G, output_path=os.path.join(out_dir, "code_graph.png"))

    print("\nAll files saved in current directory.")
