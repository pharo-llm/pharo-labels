"""Pure-Python Pharo method parser producing a JSON-serialisable AST.

This is the default ``python`` parser backend. It follows the Pharo grammar for
method source (patterns, pragmas, temporaries, statements, cascades, blocks,
literal arrays, byte arrays, dynamic arrays, scaled/radix numbers). It does not
execute code and does not resolve bindings beyond lexical scope.

The node vocabulary mirrors Pharo's RB AST (RBMethodNode, RBMessageNode, ...)
so that the optional Pharo-side exporter (``pharo/ASTExporter``) can emit the
same structure. Unsupported or malformed input raises :class:`ParseError` with
a position; callers record the failure instead of skipping it silently.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

PSEUDO_VARIABLES = {"self", "super", "thisContext"}
LITERAL_KEYWORDS = {"true": "boolean", "false": "boolean", "nil": "nil"}


class ParseError(ValueError):
    def __init__(self, message: str, position: int | None = None):
        super().__init__(message if position is None else f"{message} (offset {position})")
        self.position = position
        self.reason = message


@dataclass
class Token:
    kind: str
    value: str
    start: int
    end: int


_LEX = re.compile(
    r"(?P<space>[\s﻿]+)"
    r'|(?P<comment>"(?:[^"]|"")*")'
    r"|(?P<string>'(?:[^']|'')*')"
    r"|(?P<bytearray>\#+\[)"
    r"|(?P<literalarray>\#+\()"
    r"|(?P<brace_literal>\#\{)"
    r"|(?P<symbol>\#+(?:'(?:[^']|'')*'|[A-Za-z_][\w]*(?::[A-Za-z_][\w]*)*:*|[+\-*/\\~<>=@%|&?!,]+))"
    r"|(?P<char>\$[\s\S])"
    r"|(?P<number>\d+r-?[0-9A-Z]+(?:\.[0-9A-Z]+)?(?:e-?\d+)?|\d+(?:\.\d+)?(?:[edq]-?\d+)?(?:s\d*)?)"
    r"|(?P<assign>:=)"
    r"|(?P<keyword>[A-Za-z_]\w*:(?!=))"
    r"|(?P<id>[A-Za-z_]\w*)"
    r"|(?P<binary>[+\-*/\\~<>=@%&?,!|]+)"
    r"|(?P<punct>[\[\](){}.;:^#])"
    r"|(?P<unknown>.)",
    re.S,
)


def tokenize(source: str, keep_comments: bool = False) -> list[Token]:
    """Lex Smalltalk source. Unterminated strings/comments become ``unknown``."""
    tokens: list[Token] = []
    pos, n = 0, len(source)
    while pos < n:
        m = _LEX.match(source, pos)
        if m is None:  # pragma: no cover - the unknown branch matches anything
            raise ParseError("Lexer failure", pos)
        kind, value = m.lastgroup, m.group()
        end = m.end()
        if kind == "unknown" and value in "\"'":
            raise ParseError("Unterminated string or comment", pos)
        if kind == "binary" and len(value) > 1 and value.endswith("-") and end < n and source[end].isdigit():
            value = value[:-1]  # 3@-4 : the minus belongs to the number literal
            end -= 1
        if kind == "binary" and len(value) > 1 and "|" in value and value.strip("|") == "":
            # '||' is two temporaries/argument bars in declaration context; the
            # parser splits it where needed. Keep as one token here.
            pass
        if kind == "space":
            pos = end
            continue
        if kind == "comment" and not keep_comments:
            pos = end
            continue
        tokens.append(Token(kind, value, m.start(), end))
        pos = end
    return tokens


def line_of(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


class _Parser:
    def __init__(self, source: str):
        self.source = source
        self.tokens = tokenize(source)
        self.i = 0

    # -- token helpers -------------------------------------------------
    def peek(self, offset: int = 0) -> Token | None:
        j = self.i + offset
        return self.tokens[j] if j < len(self.tokens) else None

    def at(self, value: str | None = None, kind: str | None = None, offset: int = 0) -> bool:
        t = self.peek(offset)
        if t is None:
            return False
        return (value is None or t.value == value) and (kind is None or t.kind == kind)

    def take(self, value: str | None = None, kind: str | None = None) -> Token:
        t = self.peek()
        if value == "|" and t is not None and t.kind == "binary" and t.value.startswith("|") and len(t.value) > 1 and set(t.value) == {"|"}:
            # split '||' into two bars (block args followed by temporaries)
            self.tokens[self.i] = Token("binary", t.value[1:], t.start + 1, t.end)
            return Token("binary", "|", t.start, t.start + 1)
        if t is None or not self.at(value, kind):
            got = t.value if t else "end of source"
            raise ParseError(f"Expected {value or kind or 'token'}, got {got!r}", t.start if t else len(self.source))
        self.i += 1
        return t

    def at_bar(self) -> bool:
        t = self.peek()
        return t is not None and t.kind == "binary" and set(t.value) == {"|"}

    def node(self, node_type: str, token: Token | None, /, **fields):
        d = {"type": node_type}
        if token is not None:
            d["start"] = token.start
            d["line"] = line_of(self.source, token.start)
        d.update(fields)
        return d

    # -- method level ----------------------------------------------------
    def parse_method(self) -> dict:
        first = self.peek()
        if first is None:
            raise ParseError("Empty method source", 0)
        selector, args = self.pattern()
        pragmas: list[dict] = []
        temporaries: list[str] = []
        pragmas += self.pragmas()
        if self.at_bar():
            temporaries = self.temporaries()
        pragmas += self.pragmas()
        body = self.sequence(end=None, temporaries=temporaries)
        if self.peek() is not None:
            t = self.peek()
            raise ParseError(f"Unexpected token {t.value!r}", t.start)
        return {"type": "Method", "selector": selector, "arguments": args,
                "pragmas": pragmas, "body": body, "line": 1, "start": first.start}

    def pattern(self) -> tuple[str, list[str]]:
        t = self.peek()
        if t.kind == "id":
            self.i += 1
            return t.value, []
        if t.kind == "binary":
            self.i += 1
            return t.value, [self.take(kind="id").value]
        if t.kind == "keyword":
            parts, args = [], []
            while self.at(kind="keyword"):
                parts.append(self.take().value)
                args.append(self.take(kind="id").value)
            return "".join(parts), args
        raise ParseError(f"Invalid method pattern starting with {t.value!r}", t.start)

    def pragmas(self) -> list[dict]:
        out = []
        while self.at("<", "binary"):
            open_t = self.take()
            if self.at(kind="id"):
                sel = self.take().value
                args = []
            elif self.at(kind="keyword"):
                parts, args = [], []
                while self.at(kind="keyword"):
                    parts.append(self.take().value)
                    a = self.peek()
                    if a is None:
                        raise ParseError("Unterminated pragma", open_t.start)
                    if a.kind == "binary" and a.value == "-" and self.at(kind="number", offset=1):
                        self.i += 2
                        args.append("-" + self.tokens[self.i - 1].value)
                        continue
                    if a.kind in ("string", "symbol", "number", "char", "id", "literalarray_raw"):
                        self.i += 1
                        args.append(a.value)
                    elif a.kind == "literalarray":
                        args.append(self.literal_array(self.take())["value"])
                    else:
                        raise ParseError(f"Unsupported pragma argument {a.value!r}", a.start)
                sel = "".join(parts)
            elif self.at(kind="binary") and not self.at(">"):
                sel = self.take().value
                args = [self.take().value]
            else:
                raise ParseError("Malformed pragma", open_t.start)
            t = self.peek()
            if t is not None and t.kind == "binary" and t.value.startswith(">") and len(t.value) > 1:
                raise ParseError("Malformed pragma terminator", t.start)
            self.take(">")
            out.append({"type": "Pragma", "selector": sel, "arguments": args,
                        "line": line_of(self.source, open_t.start), "start": open_t.start})
        return out

    def temporaries(self) -> list[str]:
        self.take("|")
        names = []
        while self.at(kind="id"):
            names.append(self.take().value)
        self.take("|")
        return names

    def sequence(self, end: str | None, temporaries: list[str]) -> dict:
        statements = []
        first = self.peek()
        while self.peek() is not None and not (end and self.at(end)):
            if self.at("."):
                self.take()
                continue
            if self.at("^"):
                ret = self.take()
                statements.append(self.node("Return", ret, value=self.expression()))
            else:
                statements.append(self.expression())
            if self.at("."):
                self.take()
            elif self.peek() is not None and not (end and self.at(end)):
                t = self.peek()
                raise ParseError(f"Expected statement end, got {t.value!r}", t.start)
        return self.node("Sequence", first, temporaries=temporaries, statements=statements)

    # -- expressions -----------------------------------------------------
    def expression(self) -> dict:
        t = self.peek()
        if t is not None and t.kind == "id" and self.at(kind="assign", offset=1):
            self.i += 2
            return self.node("Assignment", t, variable=t.value, value=self.expression())
        receiver = self.primary()
        expr = self.messages(receiver)
        if self.at(";"):
            if expr["type"] != "Message":
                raise ParseError("Cascade without a message", self.peek().start)
            cascade_receiver = expr["receiver"]
            messages = [expr]
            while self.at(";"):
                self.take()
                before = self.i
                part = self.messages({"type": "CascadeReceiver"}, single=True)
                if self.i == before or part["type"] != "Message":
                    raise ParseError("Empty cascade part", self.peek().start if self.peek() else len(self.source))
                messages.append(part)
            # Later parts already chain from CascadeReceiver; the first part
            # still holds the real receiver, which moves to the Cascade node.
            messages[0]["receiver"] = {"type": "CascadeReceiver"}
            return {"type": "Cascade", "receiver": cascade_receiver, "messages": messages,
                    "line": expr.get("line"), "start": expr.get("start")}
        return expr

    def messages(self, receiver: dict, single: bool = False) -> dict:
        """Parse unary* binary* keyword? messages applied to ``receiver``.

        With ``single`` (cascade parts), the first message is sent to the
        cascade receiver and any following messages chain on its result.
        """
        expr = receiver
        while self.at(kind="id"):
            t = self.take()
            expr = self.node("Message", t, receiver=expr, selector=t.value, arguments=[], kind="unary")
        while self.at(kind="binary") and not self.at_bar_only():
            t = self.take()
            arg = self.unary_chain(self.primary())
            expr = self.node("Message", t, receiver=expr, selector=t.value, arguments=[arg], kind="binary")
        if self.at(kind="keyword"):
            first = self.peek()
            parts, args = [], []
            while self.at(kind="keyword"):
                parts.append(self.take().value)
                args.append(self.binary_chain(self.unary_chain(self.primary())))
            expr = self.node("Message", first, receiver=expr, selector="".join(parts), arguments=args, kind="keyword")
        return expr

    def at_bar_only(self) -> bool:
        # '|' as a message is legal (boolean or); bars are only special in
        # declarations, which are handled before statements are parsed.
        return False

    def unary_chain(self, expr: dict) -> dict:
        while self.at(kind="id"):
            t = self.take()
            expr = self.node("Message", t, receiver=expr, selector=t.value, arguments=[], kind="unary")
        return expr

    def binary_chain(self, expr: dict) -> dict:
        while self.at(kind="binary"):
            t = self.take()
            arg = self.unary_chain(self.primary())
            expr = self.node("Message", t, receiver=expr, selector=t.value, arguments=[arg], kind="binary")
        return expr

    def primary(self) -> dict:
        t = self.peek()
        if t is None:
            raise ParseError("Unexpected end of source; expected an expression", len(self.source))
        k = t.kind
        if k == "id":
            self.i += 1
            if t.value in LITERAL_KEYWORDS:
                return self.node("Literal", t, kind=LITERAL_KEYWORDS[t.value], value=t.value)
            return self.node("Variable", t, name=t.value)
        if k == "string":
            self.i += 1
            return self.node("Literal", t, kind="string", value=t.value[1:-1].replace("''", "'"))
        if k == "symbol":
            self.i += 1
            v = t.value.lstrip("#")
            if v.startswith("'"):
                v = v[1:-1].replace("''", "'")
            return self.node("Literal", t, kind="symbol", value=v)
        if k == "char":
            self.i += 1
            return self.node("Literal", t, kind="character", value=t.value[1:])
        if k == "number":
            self.i += 1
            return self.node("Literal", t, kind="number", value=t.value)
        if k == "binary" and t.value == "-" and self.at(kind="number", offset=1) and self.peek(1).start == t.end:
            self.i += 2
            return self.node("Literal", t, kind="number", value="-" + self.tokens[self.i - 1].value)
        if k == "literalarray":
            return self.literal_array(self.take())
        if k == "bytearray":
            open_t = self.take()
            values = []
            while not self.at("]"):
                if not self.at(kind="number"):
                    t2 = self.peek()
                    raise ParseError("Invalid byte array element", t2.start if t2 else len(self.source))
                values.append(self.take().value)
            self.take("]")
            return self.node("Literal", open_t, kind="bytearray", value=" ".join(values))
        if k == "brace_literal":
            raise ParseError("Unsupported binding literal #{...}", t.start)
        if t.value == "(":
            self.take()
            e = self.expression()
            self.take(")")
            return e
        if t.value == "[":
            return self.block()
        if t.value == "{":
            open_t = self.take()
            seq = self.sequence(end="}", temporaries=[])
            self.take("}")
            return self.node("DynamicArray", open_t, statements=seq["statements"])
        raise ParseError(f"Unexpected token {t.value!r}; expected an expression", t.start)

    def literal_array(self, open_t: Token) -> dict:
        depth, start = 1, open_t.start
        while depth:
            t = self.peek()
            if t is None:
                raise ParseError("Unterminated literal array", open_t.start)
            self.i += 1
            if t.value in ("(", "#(") or t.kind == "literalarray":
                depth += 1
            elif t.value == ")":
                depth -= 1
            elif t.kind == "bytearray":
                while not self.at("]"):
                    if self.peek() is None:
                        raise ParseError("Unterminated byte array", t.start)
                    self.i += 1
                self.i += 1
        end = self.tokens[self.i - 1].end
        return self.node("Literal", open_t, kind="array", value=self.source[start:end])

    def block(self) -> dict:
        open_t = self.take("[")
        args = []
        while self.at(":"):
            self.take(":")
            args.append(self.take(kind="id").value)
        if args and not self.at("]"):
            t = self.peek()
            if t is not None and t.kind == "binary" and t.value == "||":
                # [:x || t | ...] = end of args followed by temporaries
                self.take("|")
            else:
                self.take("|")
        temporaries = []
        if self.at_bar():
            temporaries = self.temporaries()
        body = self.sequence(end="]", temporaries=temporaries)
        self.take("]")
        return self.node("Block", open_t, arguments=args, body=body)


def parse_method(source: str) -> dict:
    """Parse one method (pattern + body) and return its AST dictionary."""
    return _Parser(source).parse_method()


def parse_expression_sequence(source: str) -> dict:
    """Parse a doit/script body (no method pattern)."""
    p = _Parser(source)
    temps = p.temporaries() if p.at_bar() else []
    seq = p.sequence(end=None, temporaries=temps)
    return {"type": "Method", "selector": "DoIt", "arguments": [], "pragmas": [], "body": seq, "line": 1, "start": 0}


def iter_nodes(node, parent=None, depth=0, block_depth=0):
    """Yield (node, parent, depth, block_depth) for every AST node, pre-order."""
    if not isinstance(node, dict):
        return
    yield node, parent, depth, block_depth
    t = node.get("type")
    child_block_depth = block_depth + (1 if t == "Block" else 0)
    for key in ("body", "receiver", "value"):
        child = node.get(key)
        if isinstance(child, dict):
            yield from iter_nodes(child, node, depth + 1, child_block_depth)
    for key in ("arguments", "statements", "messages"):
        children = node.get(key)
        if isinstance(children, list):
            for c in children:
                if isinstance(c, dict):
                    yield from iter_nodes(c, node, depth + 1, child_block_depth)


def count_nodes(ast: dict) -> int:
    return sum(1 for _ in iter_nodes(ast))
