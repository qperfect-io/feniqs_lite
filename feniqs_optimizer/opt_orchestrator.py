# Copyright © 2025 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
#!/usr/bin/env python3
"""
opt_orchestrator.py

Orchestration script for external quantum circuit optimizers.

Overview
--------
This program acts as an orchestrator for multiple external quantum circuit
optimization tools that operate on OpenQASM input files. It does not implement
quantum rewrite rules internally. Instead, it manages:

- QASM parsing and reconstruction,
- chunk-based orchestration for selected optimizers,
- external tool execution,
- acceptance or rejection of optimizer output based on a cost metric,
- writing the final optimized circuit.

Supported optimizers
--------------------
1. staq
   - Applied in chunk-based mode on unitary chunks extracted from the circuit.

2. zx (PyZX CLI)
   - Applied in chunk-based mode on unitary chunks extracted from the circuit.
   - Includes QASM sanitation helpers to improve compatibility with PyZX.

3. feynopt
   - Applied to the entire input QASM file as a whole.
   - No chunking is used.
   - The optimization mode is fixed to O4.

Design philosophy
-----------------
Different external optimizers often work best under different orchestration
strategies.

- staq and zx are used here in a chunk-oriented workflow, where the circuit is
  split into unitary pieces and optimized incrementally - very useful for large circuits.

- feynopt is treated as a whole-file optimizer. The full QASM file is passed to
  feynopt directly, without segmenting the circuit into unitary regions.

This script therefore serves as an orchestration layer that applies different
execution strategies depending on the optimizer.

Cost metric
-----------
Optimizer results are evaluated using a simple weighted metric:

    cost = T_count + 10 * CX_count

where:
- T_count counts both T and TDG gates,
- CX_count counts CX gates.

This metric gives greater weight to CX reduction, which is often more valuable
than reducing single-qubit phase gates.

External dependencies
---------------------
Depending on the selected optimizer, the following tools are expected to be
available in PATH:

- staq
- python
- pyzx (used through "python -m pyzx")
- feynopt

Typical usage
-------------
staq:
    python qasm_orchestrator.py input.qasm -o out.qasm \
        --optimizer staq --chunk-size 200 --timeout 120

zx:
    python qasm_orchestrator.py input.qasm -o out.qasm \
        --optimizer zx --chunk-size 200 --timeout 120

feynopt:
    python qasm_orchestrator.py input.qasm -o out.qasm \
        --optimizer feynopt --chunk-size 1 --timeout 300

Notes
-----
- --chunk-size is required by the command-line interface because it is used for
  staq and zx. It is ignored when optimizer=feynopt.
- feynopt always runs in O4 mode in this script.
- This program is an orchestrator for different quantum circuit optimizers.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple



# Structured process result
@dataclass
class RunResult:
    """
    Structured result of an external process execution.

    Attributes
    ----------
    ok : bool
        True if the process exited successfully.
    stdout : str
        Captured standard output.
    stderr : str
        Captured standard error.
    returncode : int
        Process return code. A timeout is represented as -1.
    elapsed_s : float
        Execution time in seconds.
    reason : str
        Optional reason string such as "timeout" or "nonzero".
    """
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    elapsed_s: float
    reason: str = ""



# QASM parsing and syntax helpers
QREG = re.compile(r"qreg\s+\w+\[(\d+)\]\s*;", re.IGNORECASE)
NON_UNITARY_PREFIX = ("measure", "reset", "if(", "creg")

_U_BROKEN = re.compile(
    r"^\s*(u[123])\s+(.+?)\s+(q\[\d+\])\s*;\s*$",
    re.IGNORECASE
)

_GATE_WITH_PARAMS = re.compile(
    r"\b(rz|u1|u2|u3)\s*\(\s*([^)]+)\s*\)",
    re.IGNORECASE
)

UNSUPPORTED_PYZX_PHASE = re.compile(
    r"rz\s*\([^)]*pi[^)]*\)",
    re.IGNORECASE
)


def _safe_eval_expr(expr: str) -> float:
    """
    Safely evaluate a restricted arithmetic expression that may contain pi.

    Supported syntax
    ----------------
    - numeric constants
    - pi
    - unary + and -
    - binary +, -, *, /

    Parameters
    ----------
    expr : str
        Arithmetic expression.

    Returns
    -------
    float
        Numeric value of the expression.
    """
    node = ast.parse(expr, mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id == "pi":
            return math.pi
        if isinstance(n, ast.UnaryOp):
            value = ev(n.operand)
            return value if isinstance(n.op, ast.UAdd) else -value
        if isinstance(n, ast.BinOp):
            left = ev(n.left)
            right = ev(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
        raise ValueError(expr)

    return float(ev(node))


def _split_top_level_commas(text: str) -> List[str]:
    """
    Split a parameter list by commas that are not inside nested parentheses.

    Parameters
    ----------
    text : str
        Raw parameter list.

    Returns
    -------
    list[str]
        Top-level parameter elements.
    """
    out, cur, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur).strip())
    return out


def sanitize_for_pyzx(qasm: str, decimals: int = 12) -> str:
    """
    Make QASM more robust for PyZX parsing.

    This function performs conservative sanitation:
    - repairs broken u1/u2/u3 syntax,
    - converts symbolic parameters to numeric floating-point values where
      possible,
    - leaves lines unchanged if safe conversion is not possible.

    Parameters
    ----------
    qasm : str
        Input QASM text.
    decimals : int, optional
        Number of significant digits used for formatted numeric parameters.

    Returns
    -------
    str
        Sanitized QASM text.
    """
    out_lines: List[str] = []

    for line in qasm.splitlines(True):
        ln = line.rstrip("\n")

        m = _U_BROKEN.match(ln)
        if m:
            gate, params, qubit = m.groups()
            ln = f"{gate}({params}) {qubit};"

        s = ln.strip().lower()
        if not s.startswith(("rz(", "u1(", "u2(", "u3(")):
            out_lines.append(ln + "\n")
            continue

        gate = ln.split("(", 1)[0].strip()

        p0 = ln.find("(")
        depth = 0
        p1 = None
        for i in range(p0, len(ln)):
            if ln[i] == "(":
                depth += 1
            elif ln[i] == ")":
                depth -= 1
                if depth == 0:
                    p1 = i
                    break

        if p1 is None:
            out_lines.append(ln + "\n")
            continue

        param_str = ln[p0 + 1:p1]
        rest = ln[p1 + 1:]

        try:
            parts = _split_top_level_commas(param_str)
            vals = [_safe_eval_expr(p) for p in parts]
            numeric = ",".join(f"{v:.{decimals}g}" for v in vals)
            ln = f"{gate}({numeric}){rest}"
        except Exception:
            pass

        out_lines.append(ln + "\n")

    return "".join(out_lines)


def normalize_all_gate_parameters_for_pyzx(qasm: str, decimals: int = 12) -> str:
    """
    Normalize rz/u1/u2/u3 gate parameters to numeric values.

    This helper is intentionally defensive because some external tools may emit
    imperfect parameter syntax.

    Parameters
    ----------
    qasm : str
        Input QASM text.
    decimals : int, optional
        Number of significant digits used for formatted numeric parameters.

    Returns
    -------
    str
        QASM with normalized gate parameters.
    """
    def clean_expr(expr: str) -> str:
        expr = expr.replace("(", "").replace(")", "")
        return expr.strip()

    def repl(match: re.Match) -> str:
        gate = match.group(1)
        params = match.group(2)

        parts = [p.strip() for p in params.split(",")]
        out = []

        for p in parts:
            p = clean_expr(p)
            try:
                value = eval(p, {"__builtins__": {}}, {"pi": math.pi})
            except Exception:
                return match.group(0)
            out.append(f"{value:.{decimals}g}")

        return f"{gate}({','.join(out)})"

    return _GATE_WITH_PARAMS.sub(repl, qasm)


def enforce_space_before_qubit(qasm: str) -> str:
    """
    Ensure a space exists between a parameterized gate and its target qubit.

    Example
    -------
    rz(0.1)q[3]  ->  rz(0.1) q[3]

    Parameters
    ----------
    qasm : str
        Input QASM text.

    Returns
    -------
    str
        Reformatted QASM text.
    """
    return re.sub(r"\)(\s*)(q\[)", r") \2", qasm)


def pyzx_safe(qasm: str) -> bool:
    """
    Check whether the QASM appears safe enough for the PyZX workflow.

    In this orchestrator, symbolic pi expressions inside rz(...) are treated as
    potentially unsafe and should be normalized first.

    Parameters
    ----------
    qasm : str
        Input QASM text.

    Returns
    -------
    bool
        True if the QASM appears suitable for PyZX processing.
    """
    return not bool(UNSUPPORTED_PYZX_PHASE.search(qasm))


def is_non_unitary_line(line: str) -> bool:
    """
    Determine whether a QASM line starts a non-unitary instruction.

    Parameters
    ----------
    line : str
        QASM source line.

    Returns
    -------
    bool
        True if the line is treated as a non-unitary boundary.
    """
    s = line.strip().lower()
    if not s or s.startswith("//"):
        return False
    return s.startswith(NON_UNITARY_PREFIX)


def split_header_and_body(qasm: str) -> Tuple[str, List[str]]:
    """
    Split QASM into header and body.

    The header contains declarations and setup lines that must be preserved.
    The body contains gate-level instructions that may be chunked for some
    optimizers.

    Parameters
    ----------
    qasm : str
        Full QASM text.

    Returns
    -------
    tuple[str, list[str]]
        Header text and body lines.
    """
    lines = qasm.splitlines(True)
    header: List[str] = []
    body: List[str] = []
    seen_qreg = False

    for ln in lines:
        if QREG.search(ln):
            seen_qreg = True
            header.append(ln)
            continue
        if not seen_qreg:
            header.append(ln)
        else:
            body.append(ln)

    if not any(QREG.search(l) for l in header):
        header = lines[:3]
        body = lines[3:]

    return "".join(header), body


def build_qasm(header: str, body_lines: List[str]) -> str:
    """
    Reconstruct a complete QASM document from header and body lines.

    Parameters
    ----------
    header : str
        QASM header.
    body_lines : list[str]
        Body lines.

    Returns
    -------
    str
        Full reconstructed QASM text.
    """
    if header and not header.endswith("\n"):
        header += "\n"
    return header + "".join(body_lines)


def segment_body(body_lines: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Divide the QASM body into unitary and raw segments.

    This segmentation is used only for chunk-based optimizers such as staq and
    zx. It is not used for feynopt in this version of the script.

    Parameters
    ----------
    body_lines : list[str]
        QASM body lines.

    Returns
    -------
    list[tuple[str, list[str]]]
        Ordered segments of type ("unitary", lines) or ("raw", lines).
    """
    segments: List[Tuple[str, List[str]]] = []
    current: List[str] = []

    for ln in body_lines:
        if is_non_unitary_line(ln):
            if current:
                segments.append(("unitary", current))
                current = []
            segments.append(("raw", [ln]))
        else:
            current.append(ln)

    if current:
        segments.append(("unitary", current))

    return segments


def chunk_unitary_lines(lines: List[str], chunk_size: int) -> List[List[str]]:
    """
    Divide a unitary segment into fixed-size chunks.

    This helper is used only for staq and zx orchestration.

    Parameters
    ----------
    lines : list[str]
        Unitary QASM lines.
    chunk_size : int
        Maximum number of non-comment operations per chunk.

    Returns
    -------
    list[list[str]]
        Chunked line groups.
    """
    chunks: List[List[str]] = []
    current: List[str] = []
    count = 0

    for ln in lines:
        current.append(ln)
        s = ln.strip()
        if s and not s.startswith("//"):
            count += 1
        if count >= chunk_size:
            chunks.append(current)
            current = []
            count = 0

    if current:
        chunks.append(current)

    return chunks



# Cost metric

def count_t_cx(qasm: str) -> Tuple[int, int]:
    """
    Count T-family gates and CX gates in QASM text.

    T-family includes:
    - t
    - tdg

    Parameters
    ----------
    qasm : str
        Input QASM text.

    Returns
    -------
    tuple[int, int]
        Pair (t_count, cx_count).
    """
    t = 0
    cx = 0
    for ln in qasm.splitlines():
        s = ln.strip().lower()
        if s.startswith("t ") or s.startswith("tdg "):
            t += 1
        if s.startswith("cx "):
            cx += 1
    return t, cx


def cost_t_cx(qasm: str, w_cx: int = 10) -> int:
    """
    Compute the weighted circuit cost used by this orchestrator.

    Parameters
    ----------
    qasm : str
        Input QASM text.
    w_cx : int, optional
        Weight assigned to each CX gate.

    Returns
    -------
    int
        Weighted cost.
    """
    t, cx = count_t_cx(qasm)
    return t + w_cx * cx


# External process utilities
def require_tool(name: str) -> None:
    """
    Ensure that an external executable is available in PATH.

    Parameters
    ----------
    name : str
        Executable name.

    Raises
    ------
    SystemExit
        If the executable is not found.
    """
    if shutil.which(name) is None:
        raise SystemExit(f"ERROR: '{name}' not found in PATH.")


def run_process(cmd: List[str], timeout: int) -> RunResult:
    """
    Execute an external process with timeout and captured output.

    Parameters
    ----------
    cmd : list[str]
        Command and arguments.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    RunResult
        Structured execution result.
    """
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        elapsed = time.perf_counter() - t0
        ok = (proc.returncode == 0)
        return RunResult(
            ok=ok,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            returncode=proc.returncode,
            elapsed_s=elapsed,
            reason="" if ok else "nonzero",
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return RunResult(
            ok=False,
            stdout="",
            stderr="",
            returncode=-1,
            elapsed_s=elapsed,
            reason="timeout",
        )



# External optimizer wrappers
def run_staq(qasm_text: str, timeout: int, verbose: bool = False) -> Optional[str]:
    """
    Run staq as an external optimizer.

    Parameters
    ----------
    qasm_text : str
        Input QASM text.
    timeout : int
        Timeout in seconds.
    verbose : bool, optional
        Whether to print diagnostic information.

    Returns
    -------
    str | None
        Optimized QASM if successful, otherwise None.
    """
    require_tool("staq")

    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.qasm"
        out = Path(td) / "out.qasm"

        inp.write_text(qasm_text, encoding="utf-8")
        cmd = ["staq", "-O3", "-s", "-c", "-r", str(inp), "-o", str(out)]
        rr = run_process(cmd, timeout=timeout)

        if verbose:
            print("[staq cmd]", " ".join(cmd), "|", rr.reason, f"{rr.elapsed_s:.2f}s")
            if rr.stderr.strip():
                print("[staq stderr]", rr.stderr.strip()[:4000])

        if not rr.ok or not out.exists():
            return None

        txt = out.read_text(encoding="utf-8")
        return txt if txt.strip() else None


def run_pyzx_external(
    qasm_text: str,
    timeout: int,
    simp: str = "full",
    phase_poly: bool = False,
    verbose: bool = False
) -> Optional[str]:
    """
    Run PyZX via its command-line interface.

    Before execution, the QASM is sanitized and normalized to improve parser
    robustness.

    Parameters
    ----------
    qasm_text : str
        Input QASM text.
    timeout : int
        Timeout in seconds.
    simp : str, optional
        PyZX simplification mode.
    phase_poly : bool, optional
        Whether to enable phase-polynomial processing.
    verbose : bool, optional
        Whether to print diagnostic information.

    Returns
    -------
    str | None
        Optimized QASM if successful, otherwise None.
    """
    require_tool("python")

    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.qasm"
        out = Path(td) / "out.qasm"

        qasm_text = sanitize_for_pyzx(qasm_text)
        qasm_text = normalize_all_gate_parameters_for_pyzx(qasm_text)
        qasm_text = enforce_space_before_qubit(qasm_text)

        if not pyzx_safe(qasm_text):
            return None

        inp.write_text(qasm_text, encoding="utf-8")

        cmd = ["python", "-m", "pyzx", "opt", "-d", str(out), "-t", "qasm", "-g", simp]
        if phase_poly:
            cmd.append("-p")
        cmd.append(str(inp))

        rr = run_process(cmd, timeout)

        if verbose:
            print("[pyzx cmd]", " ".join(cmd), "|", rr.reason, f"{rr.elapsed_s:.2f}s")
            if rr.stderr.strip():
                print("[pyzx stderr]", rr.stderr.strip()[:4000])

        if not rr.ok or not out.exists():
            return None

        txt = out.read_text(encoding="utf-8")
        return txt if txt.strip() else None


def run_feynopt(qasm_text: str, timeout: int, verbose: bool = False) -> Optional[str]:
    """
    Run feynopt on the entire QASM file in fixed O4 mode.

    This function does not segment or chunk the input circuit. The full QASM
    file is passed directly to feynopt as a whole-file optimization task.

    The wrapper is intentionally robust:
    - first tries file output via -o,
    - then tries stdout,
    - then searches for any generated .qasm file in the temporary directory.

    Parameters
    ----------
    qasm_text : str
        Full input QASM text.
    timeout : int
        Timeout in seconds.
    verbose : bool, optional
        Whether to print diagnostic information.

    Returns
    -------
    str | None
        Optimized QASM if successful, otherwise None.
    """
    require_tool("feynopt")

    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.qasm"
        out = Path(td) / "out.qasm"
        inp.write_text(qasm_text, encoding="utf-8")

        cmd = ["feynopt", "-O4", str(inp), "-o", str(out)]
        rr = run_process(cmd, timeout)

        if verbose:
            print("[feynopt cmd]", " ".join(cmd), "|", rr.reason, f"{rr.elapsed_s:.2f}s")
            if rr.stderr.strip():
                print("[feynopt stderr]", rr.stderr.strip()[:4000])

        if rr.ok and out.exists():
            txt = out.read_text(encoding="utf-8")
            if txt.strip():
                return txt

        cmd2 = ["feynopt", "-O4", str(inp)]
        rr2 = run_process(cmd2, timeout)

        if rr2.ok and rr2.stdout.strip():
            return rr2.stdout

        for qf in Path(td).glob("*.qasm"):
            if qf.name != "in.qasm":
                txt = qf.read_text(encoding="utf-8")
                if txt.strip():
                    return txt

        return None


# Optimizer orchestration logic

def optimize_chunk_staq_zx(
    header: str,
    chunk_lines: List[str],
    optimizer: str,
    timeout: int,
    accept_any: bool,
    verbose: bool
) -> List[str]:
    """
    Optimize a single chunk using either staq or zx.

    Acceptance policy
    -----------------
    The optimized chunk is accepted if:
    - it reduces the weighted cost, or
    - --accept-any is enabled and the output differs from the input.

    Parameters
    ----------
    header : str
        QASM header.
    chunk_lines : list[str]
        Chunk body lines.
    optimizer : str
        Either "staq" or "zx".
    timeout : int
        Timeout in seconds.
    accept_any : bool
        Accept any changed result, even if the weighted cost does not decrease.
    verbose : bool
        Enable diagnostic logging.

    Returns
    -------
    list[str]
        Chosen chunk lines.
    """
    original_qasm = build_qasm(header, chunk_lines)
    original_cost = cost_t_cx(original_qasm)

    t0 = time.perf_counter()
    if optimizer == "zx":
        opt_qasm = run_pyzx_external(
            original_qasm,
            timeout=timeout,
            simp="full",
            phase_poly=False,
            verbose=verbose
        )
    elif optimizer == "staq":
        opt_qasm = run_staq(original_qasm, timeout=timeout, verbose=verbose)
    else:
        raise ValueError(optimizer)
    dt = time.perf_counter() - t0

    if not opt_qasm:
        if verbose:
            print(f"  [chunk] {optimizer}: no output ({dt:.3f}s)")
        return chunk_lines

    _, new_body = split_header_and_body(opt_qasm)
    new_cost = cost_t_cx(opt_qasm)

    if (accept_any and opt_qasm != original_qasm) or (new_cost < original_cost):
        if verbose:
            t0c, cx0c = count_t_cx(original_qasm)
            t1c, cx1c = count_t_cx(opt_qasm)
            print(
                f"  [chunk] {optimizer}: ACCEPT cost {original_cost}->{new_cost} "
                f"(T {t0c}->{t1c}, CX {cx0c}->{cx1c}) ({dt:.3f}s)"
            )
        return new_body

    if verbose:
        print(f"  [chunk] {optimizer}: REJECT ({dt:.3f}s)")
    return chunk_lines


def optimize_whole_file_feynopt(
    qasm: str,
    timeout: int,
    accept_any: bool,
    verbose: bool
) -> str:
    """
    Optimize the entire QASM file using feynopt in fixed O4 mode.

    This is the whole-file orchestration path for feynopt:
    - no segmentation,
    - no chunking,
    - no ML preprocessing,
    - one direct call to feynopt on the full input file.

    Acceptance policy
    -----------------
    The optimizer output is accepted if:
    - it reduces the weighted cost, or
    - --accept-any is enabled and the output differs from the input.

    Parameters
    ----------
    qasm : str
        Full input QASM text.
    timeout : int
        Timeout in seconds.
    accept_any : bool
        Accept any changed result, even if the weighted cost does not decrease.
    verbose : bool
        Enable diagnostic logging.

    Returns
    -------
    str
        Final chosen QASM text.
    """
    base_cost = cost_t_cx(qasm)
    t0c, cx0c = count_t_cx(qasm)

    out = run_feynopt(qasm, timeout=timeout, verbose=verbose)
    if not out:
        if verbose:
            print("[feynopt] no output -> keep original file")
        return qasm

    new_cost = cost_t_cx(out)
    t1c, cx1c = count_t_cx(out)

    if (accept_any and out != qasm) or (new_cost < base_cost):
        if verbose:
            print(
                f"[feynopt] ACCEPT whole-file optimization: cost {base_cost}->{new_cost} "
                f"(T {t0c}->{t1c}, CX {cx0c}->{cx1c})"
            )
        return out

    if verbose:
        print(
            f"[feynopt] REJECT whole-file optimization: cost {base_cost}->{new_cost} "
            f"(T {t0c}->{t1c}, CX {cx0c}->{cx1c})"
        )
    return qasm


def optimize_qasm(
    qasm: str,
    chunk_size: int,
    optimizer: str,
    timeout: int,
    accept_any: bool,
    verbose: bool
) -> str:
    """
    Top-level optimization dispatcher.

    Dispatch strategy
    -----------------
    - feynopt:
        optimize the entire file in one call, without segmentation or chunking.
    - staq / zx:
        use chunk-based optimization on unitary parts of the circuit.

    Parameters
    ----------
    qasm : str
        Full input QASM text.
    chunk_size : int
        Chunk size for staq/zx. Ignored for feynopt.
    optimizer : str
        One of: "staq", "zx", "feynopt".
    timeout : int
        Timeout per external optimizer call in seconds.
    accept_any : bool
        Accept changed output even without cost improvement.
    verbose : bool
        Enable diagnostic logging.

    Returns
    -------
    str
        Optimized QASM text.
    """
    if optimizer == "feynopt":
        return optimize_whole_file_feynopt(
            qasm=qasm,
            timeout=timeout,
            accept_any=accept_any,
            verbose=verbose
        )

    header, body = split_header_and_body(qasm)
    segments = segment_body(body)

    all_chunks = []
    for typ, lines in segments:
        if typ == "unitary":
            all_chunks.extend(chunk_unitary_lines(lines, chunk_size))
    total_chunks = len(all_chunks)

    out_body: List[str] = []
    chunk_index = 0

    for typ, lines in segments:
        if typ == "raw":
            out_body.extend(lines)
            continue

        for chunk in chunk_unitary_lines(lines, chunk_size):
            chunk_index += 1
            if verbose:
                print(f"[progress] chunk {chunk_index}/{total_chunks}")

            out_body.extend(
                optimize_chunk_staq_zx(
                    header=header,
                    chunk_lines=chunk,
                    optimizer=optimizer,
                    timeout=timeout,
                    accept_any=accept_any,
                    verbose=verbose
                )
            )

    return build_qasm(header, out_body)



# Command-line interface

def main() -> None:
    """
    Parse command-line arguments, run the selected optimizer orchestration
    strategy, and write the resulting QASM file.

    Parameters
    ----------
    input : str
        Path to the input OpenQASM file.

    -o / --output : str
        Path to the output OpenQASM file.

    --optimizer : {"zx", "feynopt", "staq"}
        External optimizer to use.

    --chunk-size : int
        Chunk size for staq/zx.
        Ignored when optimizer=feynopt.

    --timeout : int
        Timeout in seconds for each external optimizer call.

    --accept-any : flag
        Accept changed output even if the weighted cost does not decrease.

    --verbose : flag
        Enable detailed logging.
    """
    ap = argparse.ArgumentParser(
        description="Orchestrator for external quantum circuit optimizers on OpenQASM."
    )
    ap.add_argument("input", help="Input OpenQASM file.")
    ap.add_argument("-o", "--output", required=True, help="Output OpenQASM file.")
    ap.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Chunk size for staq/zx. Ignored for feynopt."
    )
    ap.add_argument(
        "--optimizer",
        choices=["zx", "feynopt", "staq"],
        required=True,
        help="External optimizer to use."
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per external optimizer call in seconds."
    )
    ap.add_argument(
        "--accept-any",
        action="store_true",
        help="Accept any changed output even without cost improvement."
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging."
    )

    args = ap.parse_args()

    qasm = Path(args.input).read_text(encoding="utf-8", errors="ignore")

    t0 = time.perf_counter()
    out = optimize_qasm(
        qasm=qasm,
        chunk_size=args.chunk_size,
        optimizer=args.optimizer,
        timeout=args.timeout,
        accept_any=args.accept_any,
        verbose=args.verbose,
    )
    dt = time.perf_counter() - t0

    Path(args.output).write_text(out, encoding="utf-8")

    in_cost = cost_t_cx(qasm)
    out_cost = cost_t_cx(out)
    t_in, cx_in = count_t_cx(qasm)
    t_out, cx_out = count_t_cx(out)

    print(
        f"[summary] optimizer={args.optimizer} time={dt:.2f}s "
        f"cost {in_cost}->{out_cost} "
        f"(T {t_in}->{t_out}, CX {cx_in}->{cx_out})"
    )


if __name__ == "__main__":
    main()





