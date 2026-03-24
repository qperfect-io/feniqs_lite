#
# Copyright © 2026 QPerfect. All Rights Reserved.
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

from __future__ import annotations
import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

QREG_RE = re.compile(r"^\s*qreg\s+([a-zA-Z_]\w*)\[(\d+)\]\s*;\s*$", re.IGNORECASE)
CREG_RE = re.compile(r"^\s*creg\s+([a-zA-Z_]\w*)\[(\d+)\]\s*;\s*$", re.IGNORECASE)
QOP_RE = re.compile(r"([a-zA-Z_]\w*)\[(\d+)\]")
GATE_RE = re.compile(r"^\s*([a-zA-Z_][\w]*)\b", re.IGNORECASE)
PARAM_RE = re.compile(r"^[a-zA-Z_][\w]*\((.*?)\)")
NON_UNITARY_PREFIX = ("measure", "barrier", "creg", "if(", "reset", "opaque")
CONTROLLED_GATES = {"cp", "cx", "cz", "crx", "cry", "crz", "cu1", "ccx"}
ENTANGLING_GATES = {"cx", "cz", "swap", "iswap", "ecr", "cp", "crx", "cry", "crz", "rxx", "ryy", "rzz"}
CLIFFORD_GATES = {"id", "x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap"}
PHASE_GATES = {"s", "sdg", "t", "tdg", "rz", "p", "u1", "cp", "crz", "rzz"}
ROTATION_GATES = {"rx", "ry", "rz", "p", "u", "u1", "u2", "u3", "rxx", "ryy", "rzz", "crx", "cry", "crz", "cp"}
COMMON_GATES = ["cx","cz","swap","h","x","y","z","s","sdg","t","tdg","rx","ry","rz","u","u1","u2","u3","p","cp","crx","cry","crz"]

PI = math.pi

def normalize_qasm(qasm: str) -> str:
    out = []
    for ln in qasm.splitlines():
        s = ln.strip()
        if not s or s.startswith("//"):
            continue
        out.append(s)
    return "\n".join(out) + "\n"

def qasm_sha256(qasm: str) -> str:
    return hashlib.sha256(normalize_qasm(qasm).encode("utf-8")).hexdigest()

def read_qasm(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

def _lg(x: float) -> float:
    return math.log1p(max(0.0, x))

def _entropy_from_counts(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log(p + 1e-15)
    return ent

def _gini(values: List[float]) -> float:
    arr = sorted(float(v) for v in values if v >= 0)
    n = len(arr)
    s = sum(arr)
    if n == 0 or s <= 0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(arr))
    return (2.0 * weighted) / (n * s) - (n + 1) / n

def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mu = sum(values) / len(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))

def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = (len(vals) - 1) * q
    lo = int(math.floor(idx)); hi = int(math.ceil(idx))
    if lo == hi:
        return float(vals[lo])
    a = idx - lo
    return float((1-a)*vals[lo] + a*vals[hi])

def _approx_is_power2_pi(angle: float, tol: float = 1e-6) -> int:
    aa = abs(angle)
    if aa < tol:
        return 0
    ratio = PI / aa
    if ratio < 1 - tol:
        return 0
    k = round(math.log2(ratio)) if ratio > 0 else 0
    return int(abs((2 ** k) - ratio) < 1e-4)

def _parse_params(line: str) -> List[float]:
    m = PARAM_RE.match(line.strip())
    if not m:
        return []
    body = m.group(1)
    if not body.strip():
        return []
    vals = []
    for tok in body.split(','):
        expr = tok.strip().replace('pi', f'({PI})')
        try:
            vals.append(float(eval(expr, {"__builtins__": {}}, {})))
        except Exception:
            pass
    return vals

def is_comment_or_empty(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("//")

def is_non_unitary_line(line: str) -> bool:
    s = line.strip().lower()
    return any(s.startswith(p) for p in NON_UNITARY_PREFIX)

def build_qreg_layout(lines: Iterable[str]) -> Tuple[Dict[str, int], Dict[Tuple[str, int], int], Dict[str, int], int]:
    reg_sizes: Dict[str, int] = {}
    reg_order: List[str] = []
    c_total = 0
    for ln in lines:
        m = QREG_RE.match(ln)
        if m:
            reg_sizes[m.group(1)] = int(m.group(2)); reg_order.append(m.group(1))
        mc = CREG_RE.match(ln)
        if mc:
            c_total += int(mc.group(2))
    offsets: Dict[str, int] = {}
    cur = 0
    for r in reg_order:
        offsets[r] = cur; cur += reg_sizes[r]
    gmap = {}
    for r in reg_order:
        for i in range(reg_sizes[r]):
            gmap[(r, i)] = offsets[r] + i
    return reg_sizes, gmap, offsets, c_total

def extract_qasm_features(qasm: str) -> Dict[str, float]:
    lines = qasm.splitlines()
    reg_sizes, gmap, offsets, c_total = build_qreg_layout(lines)
    n_total = float(sum(reg_sizes.values())) if reg_sizes else 0.0
    n_maxreg = float(max(reg_sizes.values())) if reg_sizes else 0.0
    n_regs = float(len(reg_sizes))
    singleton_regs = sum(1 for s in reg_sizes.values() if s == 1)

    counts = Counter(); gate_bigrams = Counter(); gate_trigrams = Counter(); pair_counts = Counter(); pair_bigrams = Counter(); edge_counts = Counter()
    reg_pair_counts = Counter(); reg_pair_dir_counts = Counter()
    ops = ops_1q = ops_2q = ops_3qplus = 0
    non_u = measure_count = reset_count = barrier_count = conditional_count = 0
    circuit_depth = 0; prev_gate = None; prev2_gate = None
    multi_qubit_gate_count = entangling_count = controlled_count = 0
    clifford_count = phase_count = rotation_count = nonclifford_count = h_count = 0
    active_qubits = set(); active_regs = set()
    long_range_count = ultra_nonlocal_count = nearest_neighbor_count = 0
    line_end = [0] * int(n_total); qubit_depth = [0] * int(n_total)
    first_touch = [None] * int(n_total); last_touch = [None] * int(n_total)
    control_use = Counter(); target_use = Counter(); qubit_as_control = Counter(); qubit_as_target = Counter()
    gate_positions = []; phase_positions = []; pair_seq = []; late_twoq_hits = 0
    angle_abs = []; theta_vals = []; phi_vals = []; lam_vals = []; angle_bins = Counter(); power2_angles = 0; small_angles = 0; signed_angles = []
    param_gate_count = 0

    def reg_of(q: int) -> str:
        for r, off in offsets.items():
            if off <= q < off + reg_sizes[r]:
                return r
        return 'unknown'

    for pos, ln in enumerate(lines):
        if is_comment_or_empty(ln):
            continue
        s = ln.strip().lower()
        if s.startswith('measure'):
            non_u += 1; measure_count += 1; continue
        if s.startswith('reset'):
            non_u += 1; reset_count += 1; continue
        if s.startswith('barrier'):
            non_u += 1; barrier_count += 1; continue
        if s.startswith('if('):
            non_u += 1; conditional_count += 1; continue
        if is_non_unitary_line(ln):
            non_u += 1; continue
        m_gate = GATE_RE.match(ln)
        if not m_gate:
            continue
        gname = m_gate.group(1).lower()
        params = _parse_params(ln)
        if params:
            param_gate_count += 1
            for a in params:
                angle_abs.append(abs(a)); signed_angles.append(a)
                angle_bins[round(a, 6)] += 1
                if abs(a) < 1e-3: small_angles += 1
                if _approx_is_power2_pi(a): power2_angles += 1
            if len(params) >= 1: theta_vals.append(params[0])
            if len(params) >= 2: phi_vals.append(params[1])
            if len(params) >= 3: lam_vals.append(params[2])
        operands = [(r, int(i)) for (r, i) in QOP_RE.findall(ln)]
        if not operands:
            continue
        qbs = []
        qregs_local = []
        for r, i in operands:
            if (r, i) in gmap:
                q = gmap[(r, i)]
                qbs.append(q); qregs_local.append(r)
                active_qubits.add(q); active_regs.add(r)
        if not qbs:
            continue
        counts[gname] += 1; ops += 1; gate_positions.append(pos)
        if prev_gate is not None: gate_bigrams[(prev_gate, gname)] += 1
        if prev2_gate is not None and prev_gate is not None: gate_trigrams[(prev2_gate, prev_gate, gname)] += 1
        prev2_gate, prev_gate = prev_gate, gname
        if gname in CONTROLLED_GATES: controlled_count += 1
        if gname in ENTANGLING_GATES: entangling_count += 1
        if gname in CLIFFORD_GATES: clifford_count += 1
        else: nonclifford_count += 1
        if gname in PHASE_GATES: phase_count += 1; phase_positions.append(pos)
        if gname in ROTATION_GATES: rotation_count += 1
        if gname == 'h': h_count += 1
        uniq_qbs = sorted(set(qbs))
        if len(uniq_qbs) == 1:
            ops_1q += 1
        elif len(uniq_qbs) == 2:
            ops_2q += 1; multi_qubit_gate_count += 1
            u, v = uniq_qbs; a, b = (u, v) if u < v else (v, u)
            edge_counts[(a, b)] += 1; pair_counts[(a, b)] += 1; pair_seq.append((a, b))
            if len(pair_seq) >= 2: pair_bigrams[(pair_seq[-2], pair_seq[-1])] += 1
            span = abs(a - b)
            if span == 1: nearest_neighbor_count += 1
            if span >= max(2, int(n_total // 4)): long_range_count += 1
            if span >= max(2, int(n_total // 2)): ultra_nonlocal_count += 1
            if gname in CONTROLLED_GATES:
                ctrl, tgt = qbs[0], qbs[1]
                control_use[ctrl] += 1; target_use[tgt] += 1; qubit_as_control[ctrl] += 1; qubit_as_target[tgt] += 1
                rc, rt = reg_of(ctrl), reg_of(tgt)
                reg_pair_dir_counts[(rc, rt)] += 1
                if rc != rt:
                    reg_pair_counts[tuple(sorted((rc, rt)))] += 1
            else:
                ru, rv = reg_of(u), reg_of(v)
                if ru != rv:
                    reg_pair_counts[tuple(sorted((ru, rv)))] += 1
            if pos > 0.8 * len(lines): late_twoq_hits += 1
        else:
            ops_3qplus += 1; multi_qubit_gate_count += 1
            for i in range(len(uniq_qbs)):
                for j in range(i+1, len(uniq_qbs)):
                    a, b = uniq_qbs[i], uniq_qbs[j]
                    edge_counts[(a,b)] += 1; pair_counts[(a,b)] += 1; pair_seq.append((a,b))
                    if len(pair_seq) >= 2: pair_bigrams[(pair_seq[-2], pair_seq[-1])] += 1
        level = 1 + max((line_end[q] for q in uniq_qbs), default=0)
        for q in uniq_qbs:
            line_end[q] = level; qubit_depth[q] += 1
            if first_touch[q] is None: first_touch[q] = level
            last_touch[q] = level
        circuit_depth = max(circuit_depth, level)

    ops = max(ops, 1)
    cx = float(counts.get('cx', 0)); t_count = float(counts.get('t',0)+counts.get('tdg',0))
    unique_edges = len(edge_counts); total_edge_uses = float(sum(edge_counts.values())); span_values = [abs(u-v) for (u,v) in edge_counts]
    if unique_edges > 0:
        deg = Counter(); weighted_deg = Counter()
        for (u,v),w in edge_counts.items():
            deg[u]+=1; deg[v]+=1; weighted_deg[u]+=w; weighted_deg[v]+=w
        deg_vals = list(deg.values()); weighted_deg_vals = list(weighted_deg.values())
        mu = sum(deg_vals)/len(deg_vals)
        degree_variance = sum((d-mu)**2 for d in deg_vals)/len(deg_vals)
        max_degree = float(max(deg_vals)); avg_degree = float(mu)
        mean_edge_span = float(sum(span_values)/len(span_values)); max_edge_span = float(max(span_values)); std_edge_span = float(_std(span_values))
        degree_gini = float(_gini(deg_vals)); weighted_degree_gini = float(_gini(weighted_deg_vals)); span_entropy = float(_entropy_from_counts(Counter(span_values)))
    else:
        degree_variance=max_degree=avg_degree=mean_edge_span=max_edge_span=std_edge_span=degree_gini=weighted_degree_gini=span_entropy=0.0
    depth = float(circuit_depth); avg_qubit_depth = float(sum(qubit_depth)/len(qubit_depth)) if qubit_depth else 0.0; max_qubit_depth = float(max(qubit_depth)) if qubit_depth else 0.0
    active_frac = _safe_div(len(active_qubits), int(n_total)) if n_total else 0.0
    gate_entropy = _entropy_from_counts(counts); bigram_entropy = _entropy_from_counts(gate_bigrams); trigram_entropy = _entropy_from_counts(gate_trigrams)
    depth_parallelism = _safe_div(ops, depth); twoq_per_depth = _safe_div(ops_2q, depth); entangling_per_depth = _safe_div(entangling_count, depth)
    nonlocal_frac = _safe_div(long_range_count, max(1, multi_qubit_gate_count)); ultra_nonlocal_frac = _safe_div(ultra_nonlocal_count, max(1, multi_qubit_gate_count)); nearest_neighbor_frac = _safe_div(nearest_neighbor_count, max(1, multi_qubit_gate_count))
    clifford_frac = _safe_div(clifford_count, ops); nonclifford_frac = _safe_div(nonclifford_count, ops); phase_frac = _safe_div(phase_count, ops); rotation_frac = _safe_div(rotation_count, ops); entangling_frac = _safe_div(entangling_count, ops)
    lifespan = [(last_touch[q]-first_touch[q]+1) for q in range(len(first_touch)) if first_touch[q] is not None and last_touch[q] is not None]
    mean_lifespan = float(sum(lifespan)/len(lifespan)) if lifespan else 0.0; lifespan_gini = float(_gini(lifespan))
    edge_reuse = _safe_div(total_edge_uses, float(unique_edges)) if unique_edges else 0.0
    denom = int(n_total)*(int(n_total)-1)/2 if n_total >= 2 else 0.0; edge_density = _safe_div(unique_edges, denom) if denom else 0.0
    phase_position_mean_norm = _safe_div(sum(phase_positions)/len(phase_positions) if phase_positions else 0.0, max(1, len(lines)-1))
    span_q50 = _quantile(span_values,0.5); span_q75=_quantile(span_values,0.75); span_q90=_quantile(span_values,0.9)
    mean_span_norm = _safe_div(mean_edge_span, max(1.0, n_total-1)); max_span_norm = _safe_div(max_edge_span, max(1.0, n_total-1)); span_q90_norm=_safe_div(span_q90,max(1.0,n_total-1))
    weighted_span_per_twoq = _safe_div(sum(abs(u-v)*w for (u,v),w in edge_counts.items()), max(1.0,total_edge_uses))
    pair_entropy = _entropy_from_counts(pair_counts); pair_bigram_entropy = _entropy_from_counts(pair_bigrams); repeat_pair_run_frac = _safe_div(sum(1 for i in range(1,len(pair_seq)) if pair_seq[i]==pair_seq[i-1]), max(1,len(pair_seq)-1))
    control_target_imbalance = abs(sum(control_use.values()) - sum(target_use.values()))/max(1.0, sum(control_use.values())+sum(target_use.values()))
    control_gini = _gini(list(control_use.values())); target_gini = _gini(list(target_use.values()))
    ent_activity = [edge_counts[p] for p in edge_counts]; ent_activity_mean=float(sum(ent_activity)/len(ent_activity)) if ent_activity else 0.0; ent_activity_gini=float(_gini(ent_activity)); ent_activity_cv=_safe_div(_std(ent_activity), ent_activity_mean)
    qubit_depth_gini=float(_gini(qubit_depth)); late_ops_frac=_safe_div(sum(1 for p in gate_positions if p>0.8*len(lines)), max(1,len(gate_positions))); late_twoq_frac=_safe_div(late_twoq_hits, max(1, ops_2q+ops_3qplus))
    long_range_controlled_frac = _safe_div(sum(edge_counts[e] for e in edge_counts if abs(e[0]-e[1]) >= max(2,int(n_total//4))), max(1.0,total_edge_uses))
    edge_use_gini=float(_gini(list(edge_counts.values()))); edge_use_cv=_safe_div(_std(list(edge_counts.values())), edge_reuse)
    cross_reg_frac = _safe_div(sum(reg_pair_counts.values()), max(1, multi_qubit_gate_count))
    singleton_control_frac = _safe_div(sum(v for q,v in qubit_as_control.items() if reg_sizes.get(reg_of(q),0)==1), max(1,sum(qubit_as_control.values())))
    singleton_target_frac = _safe_div(sum(v for q,v in qubit_as_target.items() if reg_sizes.get(reg_of(q),0)==1), max(1,sum(qubit_as_target.values())))
    reg_interaction_entropy = _entropy_from_counts(reg_pair_dir_counts)
    parameterized_frac = _safe_div(param_gate_count, ops)
    angle_entropy = _entropy_from_counts(angle_bins)
    unique_angles = float(len(angle_bins))
    abs_angle_mean = float(sum(angle_abs)/len(angle_abs)) if angle_abs else 0.0
    abs_angle_std = float(_std(angle_abs)) if angle_abs else 0.0
    theta_std = float(_std(theta_vals)) if theta_vals else 0.0
    phi_std = float(_std(phi_vals)) if phi_vals else 0.0
    lambda_std = float(_std(lam_vals)) if lam_vals else 0.0
    small_angle_frac = _safe_div(small_angles, max(1, len(angle_abs)))
    power2pi_frac = _safe_div(power2_angles, max(1, len(angle_abs)))
    angle_sign_balance = abs(sum(1 for a in signed_angles if a>0)-sum(1 for a in signed_angles if a<0))/max(1,len(signed_angles))
    reg_size_gini = _gini(list(reg_sizes.values()))

    features = {
        'n_total': n_total, 'n_maxreg': n_maxreg, 'n_regs': n_regs,
        'ops': float(ops), 'ops_1q': float(ops_1q), 'ops_2q': float(ops_2q), 'twoq_frac': _safe_div(ops_2q, ops),
        'cx': cx, 't': t_count, 't_per_cx': _safe_div(t_count, cx), 'log_ops': _lg(ops), 'log_cx': _lg(cx), 'non_unitary': float(non_u),
        'unique_edges': float(unique_edges), 'edge_density': float(edge_density), 'edge_reuse': float(edge_reuse), 'max_degree': float(max_degree), 'avg_degree': float(avg_degree), 'degree_variance': float(degree_variance),
        'mean_edge_span': float(mean_edge_span), 'max_edge_span': float(max_edge_span), 'log_unique_edges': _lg(unique_edges), 'log_edge_reuse': _lg(edge_reuse), 'log_mean_edge_span': _lg(mean_edge_span), 'log_max_edge_span': _lg(max_edge_span),
        'n_classical': float(c_total), 'reg_balance': _safe_div(n_maxreg, n_total), 'singleton_regs': float(singleton_regs), 'reg_size_gini': float(reg_size_gini), 'cross_reg_frac': float(cross_reg_frac),
        'ops_3qplus': float(ops_3qplus), 'threeqplus_frac': _safe_div(ops_3qplus, ops),
        'controlled_count': float(controlled_count), 'controlled_frac': _safe_div(controlled_count, ops),
        'entangling_count': float(entangling_count), 'phase_count': float(phase_count), 'rotation_count': float(rotation_count), 'phase_frac': float(phase_frac), 'rotation_frac': float(rotation_frac),
        'cx_per_qubit': _safe_div(cx, n_total), 'depth': depth, 'depth_per_qubit': _safe_div(depth, n_total), 'depth_parallelism': float(depth_parallelism), 'twoq_per_depth': float(twoq_per_depth), 'entangling_per_depth': float(entangling_per_depth),
        'avg_qubit_depth': float(avg_qubit_depth), 'max_qubit_depth': float(max_qubit_depth), 'active_frac': float(active_frac), 'mean_lifespan': float(mean_lifespan), 'lifespan_gini': float(lifespan_gini), 'qubit_depth_gini': float(qubit_depth_gini),
        'measure_count': float(measure_count), 'reset_count': float(reset_count), 'barrier_count': float(barrier_count), 'conditional_count': float(conditional_count),
        'degree_gini': float(degree_gini), 'weighted_degree_gini': float(weighted_degree_gini), 'std_edge_span': float(std_edge_span), 'span_entropy': float(span_entropy),
        'nearest_neighbor_frac': float(nearest_neighbor_frac), 'nonlocal_frac': float(nonlocal_frac), 'ultra_nonlocal_frac': float(ultra_nonlocal_frac), 'gate_entropy': float(gate_entropy), 'gate_bigram_entropy': float(bigram_entropy), 'gate_trigram_entropy': float(trigram_entropy),
        'span_q50': float(span_q50), 'span_q75': float(span_q75), 'span_q90': float(span_q90), 'mean_span_norm': float(mean_span_norm), 'max_span_norm': float(max_span_norm), 'span_q90_norm': float(span_q90_norm), 'weighted_span_per_twoq': float(weighted_span_per_twoq),
        'phase_position_mean_norm': float(phase_position_mean_norm), 'pair_entropy': float(pair_entropy), 'pair_bigram_entropy': float(pair_bigram_entropy), 'repeat_pair_run_frac': float(repeat_pair_run_frac),
        'control_target_imbalance': float(control_target_imbalance), 'control_gini': float(control_gini), 'target_gini': float(target_gini), 'singleton_control_frac': float(singleton_control_frac), 'singleton_target_frac': float(singleton_target_frac), 'reg_interaction_entropy': float(reg_interaction_entropy),
        'ent_activity_mean': float(ent_activity_mean), 'ent_activity_gini': float(ent_activity_gini), 'ent_activity_cv': float(ent_activity_cv), 'late_ops_frac': float(late_ops_frac), 'late_twoq_frac': float(late_twoq_frac), 'long_range_controlled_frac': float(long_range_controlled_frac), 'edge_use_gini': float(edge_use_gini), 'edge_use_cv': float(edge_use_cv),
        'parameterized_frac': float(parameterized_frac), 'unique_angles': float(unique_angles), 'angle_entropy': float(angle_entropy), 'abs_angle_mean': float(abs_angle_mean), 'abs_angle_std': float(abs_angle_std), 'theta_std': float(theta_std), 'phi_std': float(phi_std), 'lambda_std': float(lambda_std), 'small_angle_frac': float(small_angle_frac), 'power2pi_frac': float(power2pi_frac), 'angle_sign_balance': float(angle_sign_balance),
    }
    for g in COMMON_GATES:
        features[f'gate_{g}'] = float(counts.get(g,0))
        features[f'gatefrac_{g}'] = _safe_div(counts.get(g,0), ops)
    return features

extract_features = extract_qasm_features
