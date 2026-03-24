
import pandas as pd
from mps_qasm_predictor.config import ValidationConfig
from mps_qasm_predictor.validate import summarize_topk, evaluate_predictions, summarize_eval


def test_summarize_topk_monotone():
    df = pd.DataFrame({
        'qasm_path': ['a', 'a', 'a', 'b', 'b', 'b'],
        'pred_rank': [1, 2, 3, 1, 2, 3],
        'near_optimal': [0, 1, 1, 0, 0, 1],
        'is_offline_evaluable': [1, 1, 1, 1, 1, 1],
    })
    s = summarize_topk(df, ValidationConfig(), ks=(1, 3))
    assert s['top_1_near_optimal_rate'] == 0.0
    assert s['top_3_near_optimal_rate'] == 1.0


def test_evaluate_predictions_marks_missing_offline_rows():
    full = pd.DataFrame({
        'qasm_path': ['a'],
        'matrix_product_state_max_bond_dimension': [32],
        'mps_lapack': [1],
        'opt_level': [2],
        'matrix_product_state_truncation_threshold': [1e-8],
        'mps_sample_measure_algorithm': ['mps_apply_measure'],
        'runtime': [1.0],
        'fidelity': [0.99],
    })
    pred = pd.DataFrame({
        'qasm_path': ['a'],
        'matrix_product_state_max_bond_dimension': [64],
        'mps_lapack': [1],
        'opt_level': [2],
        'matrix_product_state_truncation_threshold': [1e-8],
        'mps_sample_measure_algorithm': ['mps_apply_measure'],
        'pred_rank': [1],
    })
    ev = evaluate_predictions(full, pred, ValidationConfig())
    assert int(ev.loc[0, 'is_offline_evaluable']) == 0
    assert ev.loc[0, 'validation_status'] == 'not_evaluated_offline'


def test_summarize_eval_has_tail_metrics():
    df = pd.DataFrame({
        'qasm_path': ['a','b','c'],
        'is_offline_evaluable': [1,1,1],
        'exact_match': [1,0,0],
        'near_optimal': [1,0,0],
        'runtime_ratio_to_feasible': [1.0,1.25,1.6],
        'fidelity_gap_to_best': [0.0,0.002,0.02],
    })
    s = summarize_eval(df, ValidationConfig())
    assert 'p95_runtime_ratio' in s
    assert s['runtime_fail_rate_1p2'] > 0
    assert s['fidelity_fail_rate_tol'] > 0
