"""Extract symbols from TypeScript and TSX sources with tree-sitter."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node, Parser

# why: a React component is written as a declaration, a const-bound arrow or a default export,
# and all three are places a question can point at
_DEF_NODES = {"function_declaration", "class_declaration", "method_definition"}
_ARROW_HOLDERS = {"arrow_function", "function_expression"}


@lru_cache(maxsize=2)
def _parser(dialect: str) -> Parser:
    import tree_sitter_typescript as tst
    from tree_sitter import Language, Parser

    lang = tst.language_tsx() if dialect == "tsx" else tst.language_typescript()
    return Parser(Language(lang))


def extract(rel: str, text: str) -> list[dict]:
    """Return every definition in one TypeScript or TSX file.

    Parameters
    ----------
    rel : str
        File path relative to the snapshot source root.
    text : str
        File content.

    Returns
    -------
    list of dict
        Records with `path`, `fqname`, `kind`, `start`, `end`.
    """
    dialect = "tsx" if rel.endswith(".tsx") else "ts"
    src = text.encode("utf-8")
    tree = _parser(dialect).parse(src)
    module = rel.rsplit(".", 1)[0].replace("/", ".")
    out: list[dict] = []
    _walk(tree.root_node, src, module, rel, out)
    return out


def _name_of(node: Node, src: bytes) -> str | None:
    field = node.child_by_field_name("name")
    return src[field.start_byte : field.end_byte].decode("utf-8", "replace") if field else None


def _walk(node: Node, src: bytes, prefix: str, rel: str, out: list[dict]) -> None:
    for child in node.children:
        name = None
        kind = None
        if child.type in _DEF_NODES:
            name = _name_of(child, src)
            kind = {"function_declaration": "function", "class_declaration": "class",
                    "method_definition": "method"}[child.type]
        elif child.type == "variable_declarator":
            value = child.child_by_field_name("value")
            if value is not None and value.type in _ARROW_HOLDERS:
                name = _name_of(child, src)
                kind = "function"
        if name and kind:
            fqname = f"{prefix}.{name}"
            out.append({
                "path": rel,
                "fqname": fqname,
                "kind": kind,
                "start": child.start_point[0] + 1,
                "end": child.end_point[0] + 1,
            })
            _walk(child, src, fqname, rel, out)
        else:
            _walk(child, src, prefix, rel, out)
