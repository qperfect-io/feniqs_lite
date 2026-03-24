
# MPS QASM Predictor 

## Install
```
pip installl -e .
```
## Training
Run the following command from the project root:

```bash
python -m mps_qasm_predictor.train \
  --full-eval-csv ../data/qiskit_full_db.csv \
  --qasm-root ../data/paper_data \
  --outdir ./artifacts \
  --validate \
  --default-csv ../data/Qiskit_default_results.csv
```
This command:
- loads the full evaluation database
- loads QASM circuits
- trains the predictor
- runs validation
- compares predicted configurations against Qiskit default
 -saves all outputs into ./artifacts
## Validation
 The training command above also runs validation automatically.

It produces:
- LOFO validation: whole held-out family is removed from training
-Size-based validation: unseen qubit sizes are removed from training within each family
- Predicted vs default comparison

Main validation outputs are saved in: `./artifacts/validation`

## Remake figures

To regenerate only the final figures without rerunning training or validation, run:
```
python src/mps_qasm_predictor/remake_final_plots.py \
  --lofo-summary ./artifacts/validation/lofo_summary.csv \
  --size-summary ./artifacts/validation/size_based_summary.csv \
  --lofo-eval ./artifacts/validation/lofo_eval.csv \
  --size-eval ./artifacts/validation/size_based_eval.csv \
  --lofo-default-pairs ./artifacts/validation/compare_default_lofo/predicted_vs_default_pairs.csv \
  --size-default-pairs ./artifacts/validation/compare_default_size_based/predicted_vs_default_pairs.csv \
  --outdir ./fig_final \
  --fidelity-tol 1e-3
```
This command:
- reloads saved validation CSV files
- redraws only the final publication figures
- saves the new figures into ./fig_final
