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
from .model import fit_ranker, FEATURE_SET_PRESETS, MODEL_PROFILES
from .validate import run_validation
from .compare_default import compare_predicted_vs_default


def main():
    ap=argparse.ArgumentParser(description='Train QASM-only predictor for Qiskit or Mimiq and optionally run validation.')
    ap.add_argument('--full-eval-csv', required=True)
    ap.add_argument('--best-csv', default=None, help='Optional CSV containing one best row per circuit. If omitted, the code derives best rows from --full-eval-csv using is_best=true (or a fallback heuristic for missing circuits).')
    ap.add_argument('--best-flag-col', default='is_best', help='Column in full-eval-csv that marks best rows. Also accepts variants like is-best automatically.')
    ap.add_argument('--qasm-root', required=True)
    ap.add_argument('--feature-cache-csv', default=None, help='Optional cache for extracted circuit features. Reusing it makes repeated validations much faster.')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--backend', default='auto', choices=['auto', 'qiskit', 'mimiq'])
    ap.add_argument('--feature-set', default='advanced', choices=sorted(FEATURE_SET_PRESETS.keys()))
    ap.add_argument('--model-profile', default='full', choices=sorted(MODEL_PROFILES.keys()), help='Model size for the final trained predictor.')
    ap.add_argument('--compare-feature-sets', action='store_true', help='During validation, compare basic/structural/advanced feature presets.')
    ap.add_argument('--validate', action='store_true', help='Run validation. Use --validation-modes to choose which regimes to run.')
    ap.add_argument('--validation-modes', nargs='+', default=['random', 'lofo'], choices=['random', 'lofo', 'size', 'backend'], help='Validation regimes to run.')
    ap.add_argument('--validation-tree-jobs', type=int, default=1, help='Number of parallel jobs for tree ensembles during validation. Use 1 to reduce RAM pressure.')
    ap.add_argument('--validation-model-profile', default='lite', choices=sorted(MODEL_PROFILES.keys()), help='Smaller validation-only model profile to make validation faster.')
    ap.add_argument('--validation-bootstrap', type=int, default=60, help='Bootstrap resamples used for CI in validation metrics.')
    ap.add_argument('--validation-max-k', type=int, default=5, help='Maximum top-k rank stored and evaluated during validation.')
    ap.add_argument('--validation-size-stride', type=int, default=2, help='Use every N-th size bucket in size-based validation to reduce cost.')
    ap.add_argument('--validation-max-size-splits-per-family', type=int, default=2, help='Optional cap on number of held-out sizes per family in size-based validation.')
    ap.add_argument('--validation-max-lofo-families', type=int, default=None, help='Optional cap on the number of LOFO families. Useful for quick dry runs.')
    ap.add_argument('--validation-save-rank-tables', action='store_true', help='Persist full rank tables during validation. Off by default to keep validation lighter.')
    ap.add_argument('--validation-no-pdf', action='store_true', help='Disable PDF companion files for figures.')
    ap.add_argument('--default-csv', default=None)
    ap.add_argument('--default-filter', default=None, help='Canonical default tuple filter, e.g. bond_dimension=64,entdim=8,opt_level=1,method=zipup or sample_algorithm=mps_probabilities')
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    full, best, circuit_features = load_datasets(args.full_eval_csv, args.best_csv, args.qasm_root, backend=args.backend, best_flag_col=args.best_flag_col, feature_cache_csv=args.feature_cache_csv)
    candidate_catalogue = build_candidate_catalogue(full)
    ranker = fit_ranker(full, candidate_catalogue, feature_set=args.feature_set, model_profile=args.model_profile)
    ranker.save(outdir/'model')
    backend_name = str(full['backend'].mode().iloc[0]) if 'backend' in full.columns else args.backend
    (outdir/'model'/'metadata.json').write_text(json.dumps({'backend': backend_name, 'feature_set': args.feature_set, 'feature_cache_csv': args.feature_cache_csv, 'model_profile': args.model_profile}, indent=2), encoding='utf-8')
    circuit_features.to_csv(outdir/'circuit_features.csv', index=False)
    with open(outdir/'training_summary.json','w',encoding='utf-8') as f:
        json.dump({'backend': backend_name, 'feature_set': args.feature_set, 'model_profile': args.model_profile, 'n_full_rows': int(len(full)), 'n_best_rows': int(len(best)), 'n_unique_qasm': int(full['qasm_path'].nunique()), 'n_families': int(full['family'].nunique()), 'n_candidates': int(len(candidate_catalogue))}, f, indent=2)
    if args.validate:
        validation_modes = set(args.validation_modes)
        vcfg = ValidationConfig(
            n_bootstrap=args.validation_bootstrap,
            validation_tree_n_jobs=args.validation_tree_jobs,
            validation_model_profile=args.validation_model_profile,
            max_k=args.validation_max_k,
            save_rank_tables=args.validation_save_rank_tables,
            save_pdf_figures=not args.validation_no_pdf,
            run_random_split='random' in validation_modes,
            run_lofo='lofo' in validation_modes,
            run_size_based='size' in validation_modes,
            run_backend_transfer='backend' in validation_modes,
            size_split_stride=max(1, args.validation_size_stride),
            max_size_splits_per_family=args.validation_max_size_splits_per_family,
            max_lofo_families=args.validation_max_lofo_families,
        )
        result = run_validation(args.full_eval_csv, args.best_csv, args.qasm_root, outdir/'validation', cfg=vcfg, backend=args.backend, feature_set=args.feature_set, compare_feature_sets=args.compare_feature_sets, best_flag_col=args.best_flag_col, feature_cache_csv=args.feature_cache_csv)
        try:
            if args.default_csv and not result['lofo_top1'].empty:
                compare_predicted_vs_default(result['lofo_top1'], args.default_csv, outdir/'validation'/'compare_default_lofo', backend=args.backend, full_eval_csv=args.full_eval_csv, default_filter=args.default_filter)
            if args.default_csv and not result['size_top1'].empty:
                compare_predicted_vs_default(result['size_top1'], args.default_csv, outdir/'validation'/'compare_default_size_based', backend=args.backend, full_eval_csv=args.full_eval_csv, default_filter=args.default_filter)
        except ValueError as exc:
            print(f'[default-comparison] Skipped: {exc}')

if __name__ == '__main__':
    main()
