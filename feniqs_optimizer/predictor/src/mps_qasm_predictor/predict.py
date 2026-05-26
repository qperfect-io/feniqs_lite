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
import argparse
from dataclasses import asdict
import json
from .predictor import MPSPredictor


def main():
    ap=argparse.ArgumentParser(description='Predict Qiskit Aer MPS hyperparameters from a QASM file.')
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--qasm', required=True)
    ap.add_argument('--topk', type=int, default=1)
    args=ap.parse_args()
    predictor = MPSPredictor(args.model_dir)
    ranked = predictor.rank_qasm(args.qasm)
    print(json.dumps(ranked.head(args.topk).to_dict(orient='records'), indent=2))

if __name__ == '__main__':
    main()
