
# MPS QASM Predictor 

## Install
```
pip installl -e .
```
## Training
Run the following command from the project root:

```bash
python -m mps_qasm_predictor.run_validation   --qiskit-full-eval-csv ../data/qiskit_full_db.csv   --mimiq-full-eval-csv ../data/MimiqJuliaCpu_final\ 5.csv   --qiskit-default-csv ../data/Qiskit_default_results.csv   --mimiq-default-csv ../data/MimiqJuliaCpu_default\ 3.csv   --qasm-root ../data/paper_data   --outdir output_paper_1   --feature-set advanced   --model-profile full   --validation-model-profile full   --size-split-stride 1

```
