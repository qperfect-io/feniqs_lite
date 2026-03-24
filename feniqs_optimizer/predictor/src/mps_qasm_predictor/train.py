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
import argparse, json
from pathlib import Path
from .config import ValidationConfig
from .data import load_datasets, build_candidate_catalogue
from .model import fit_ranker
from .validate import run_validation
from .compare_default import compare_predicted_vs_default


def main():
    ap=argparse.ArgumentParser(description='Train QASM-only predictor and optionally run validation.')
    ap.add_argument('--full-eval-csv', required=True)
    ap.add_argument('--best-csv', default=None)
    ap.add_argument('--qasm-root', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--validate', action='store_true', help='Run random grouped split, leave-one-family-out, and size-based validation.')
    ap.add_argument('--default-csv', default=None)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    full, best, circuit_features = load_datasets(args.full_eval_csv, args.best_csv, args.qasm_root)
    candidate_catalogue = build_candidate_catalogue(full)
    ranker = fit_ranker(full, candidate_catalogue)
    ranker.save(outdir/'model')
    circuit_features.to_csv(outdir/'circuit_features.csv', index=False)
    with open(outdir/'training_summary.json','w',encoding='utf-8') as f:
        json.dump({'n_full_rows': int(len(full)), 'n_best_rows': int(len(best)), 'n_unique_qasm': int(full['qasm_path'].nunique()), 'n_families': int(full['family'].nunique()), 'n_candidates': int(len(candidate_catalogue))}, f, indent=2)
    if args.validate:
        result = run_validation(args.full_eval_csv, args.best_csv, args.qasm_root, outdir/'validation', cfg=ValidationConfig())
        if args.default_csv:
            compare_predicted_vs_default(result['lofo_top1'], args.default_csv, outdir/'validation'/'compare_default_lofo')
            compare_predicted_vs_default(result['size_top1'], args.default_csv, outdir/'validation'/'compare_default_size_based')

if __name__ == '__main__':
    main()
