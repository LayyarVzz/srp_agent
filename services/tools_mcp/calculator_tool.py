"""calculate 工具：安全求值四则/幂/取模表达式。

WHY 不用 eval：expression 来自外部（用户经 LLM 传入），属不可信数据；
用 AST 白名单（数字常量 + 四则 + 幂 + 取模 + 一元正负 + 括号）确定性求值，
变量 / 函数调用 / 属性访问等一律拒绝。
"""

from __future__ import annotations

import ast
import operator
from typing import Any

# AST 二元运算节点 → 操作符映射（白名单）。
_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
# AST 一元运算节点 → 操作符映射（白名单）。
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
# 允许出现的全部 AST 节点（walk 全量校验，杜绝白名单外语法）。
# 注意：`ast.walk` 会遍历到运算符节点（ast.Add 等），必须一并列入白名单。
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.UAdd,
    ast.USub,
)


def _eval_node(node: ast.AST) -> float:
    """按白名单递归求值；未知节点抛 ValueError（不进 operator 映射即拒绝）。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"不支持的表达式节点：{type(node).__name__}")


def calculate(expression: str) -> str:
    """计算算术表达式并返回结果字符串（整数结果不带小数点）。

    仅接受数字与 `+ - * / ** %` 与括号；其他语法（变量 / 函数调用 / 属性访问）
    一律拒绝（AST 白名单求值，防注入）。
    """
    text = (expression or "").strip()
    if not text:
        raise ValueError("expression 为空")
    tree = ast.parse(text, mode="eval")
    if not all(isinstance(node, _ALLOWED_NODES) for node in ast.walk(tree)):
        raise ValueError("表达式包含不允许的语法")
    value = _eval_node(tree.body)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"
