from mps_qasm_predictor.features import extract_qasm_features

def test_extract_features_basic():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[1];
    h q[0];
    cx q[0],q[1];
    rz(0.5) q[1];
    u3(pi/2,0,-pi/8) q[2];
    cx q[1],q[2];
    measure q[0] -> c[0];
    """
    feats = extract_qasm_features(qasm)
    assert feats['n_total'] == 3.0
    assert feats['cx'] == 2.0
    assert feats['measure_count'] == 1.0
    assert feats['depth'] >= 3.0
    assert 'span_q90' in feats
    assert 'controlled_frac' in feats
    assert 'parameterized_frac' in feats
    assert 'power2pi_frac' in feats
