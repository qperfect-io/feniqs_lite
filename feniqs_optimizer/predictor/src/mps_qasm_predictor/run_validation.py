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
from .config import ValidationConfig
from .validate import run_validation


def main():
    ap=argparse.ArgumentParser(description='Run  validation and figure generation.')
    ap.add_argument('--full-eval-csv', required=True)
    ap.add_argument('--best-csv', default=None)
    ap.add_argument('--qasm-root', required=Tru)
    ap.add_argument('--outdir', required=True)
    args=ap.parse_args()
    result = run_validation(args.full_eval_csv, args.best_csv, args.qasm_root, args.outdir, cfg=ValidationConfig())
    print(json.dumps(result['metrics'], indent=2))

if __name__ == '__main__':
    main()
