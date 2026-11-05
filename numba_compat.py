"""numba_compat.py — Numba JIT 的可选封装。

安装了 numba 时 ``njit`` 即 ``numba.njit``；未安装或设置环境变量 ``PPVP_DISABLE_NUMBA=1`` 时，
``njit`` 退化为恒等装饰器，同一份代码以纯 Python 执行（测试中用 ``fn.py_func`` 比对两条路径逐位一致）。
"""
from __future__ import annotations

import os

HAVE_NUMBA = False
if os.environ.get("PPVP_DISABLE_NUMBA", "0") != "1":
    try:
        from numba import njit as _numba_njit  # type: ignore

        HAVE_NUMBA = True
    except ImportError:  # pragma: no cover
        HAVE_NUMBA = False


def _with_py_func(fn):
    """numba 在 NUMBA_DISABLE_JIT=1 时返回原函数（无 py_func 属性）；统一补上，保证两条路径的调用方式一致。"""
    if not hasattr(fn, "py_func"):
        fn.py_func = fn
    return fn


def njit(*args, **kwargs):
    if HAVE_NUMBA:
        if args and callable(args[0]):
            return _with_py_func(_numba_njit(*args, **kwargs))
        deco_nb = _numba_njit(*args, **kwargs)
        return lambda fn: _with_py_func(deco_nb(fn))
    if args and callable(args[0]):
        fn = args[0]
        fn.py_func = fn
        return fn

    def deco(fn):
        fn.py_func = fn
        return fn

    return deco
