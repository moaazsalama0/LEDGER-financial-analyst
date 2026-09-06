"""Deterministic safe arithmetic -- the only place numbers may be computed.

The LLM is never allowed to produce a number from memory; every `calculated`
answer must route through `calculate()`. The implementation parses the expression
into an AST and evaluates ONLY a whitelisted subset of Python syntax.
Arbitrary code (imports, attribute access, comprehensions, lambdas, ...) is rejected.
"""
import ast
import math
import operator


class UnsafeExpressionError(ValueError):
    """Raised when the expression uses syntax outside the safe whitelist."""


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_STRUCTURAL_NODES = {
    ast.Expression,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
}

_ALLOWED_NODES = _STRUCTURAL_NODES | set(_OPERATORS.keys())


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise UnsafeExpressionError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        return _OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPERATORS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise UnsafeExpressionError("Only bare function calls are allowed")
        name = node.func.id
        if name not in _FUNCTIONS:
            raise UnsafeExpressionError(f"Function not allowed: {name}")
        return _FUNCTIONS[name](*[_eval(arg) for arg in node.args])
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise UnsafeExpressionError(f"Unknown variable: {node.id}")
    raise UnsafeExpressionError(f"Unsupported syntax: {type(node).__name__}")


def calculate(expression):
    """Safely evaluate a Python math expression. Never uses eval()."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a non-empty string")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {expression!r}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeExpressionError(f"Disallowed node type: {type(node).__name__}")
    result = _eval(tree)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise UnsafeExpressionError("Expression must evaluate to a number")
    return float(result)