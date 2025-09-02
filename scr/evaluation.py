Model Performance Validation and Measurement
==========================================

This module defines functions to evaluate the effectiveness of bias mitigation
and model performance. Evaluation is critical to determine whether the
interventions applied during fine-tuning or activation modification have
actually reduced gender bias without sacrificing caption quality. The
evaluation methods provided here are intentionally simple and generic. They
should be adapted to the specifics of your dataset and research goals.

We include two types of evaluation:

1. **Gender Classification Accuracy**: Given the model's captions, we infer
   whether the caption refers to a man, a woman or is neutral and compare
   against the ground-truth gender of the image. Accuracy differences across
   genders indicate bias. The classification uses a simple keyword-based
   approach but can be replaced with a learned classifier.

2. **Caption Quality Metrics**: Compute BLEU and ROUGE scores comparing
   generated captions to human-written captions. These metrics reflect
   fluency and faithfulness. We use the ``evaluate`` library if available.

These evaluations are not exhaustive. You may also compute calibration,
fairness gap, or other fairness metrics. For detailed analyses you can
integrate more sophisticated metrics and probes.
"""

from __future__ import annotations

import re
from typing import List, Dict, Tuple, Optional

import numpy as np

try:
    import evaluate as hf_evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False


def infer_gender_from_caption(caption: str) -> str:
    """Infer gender from a caption using keyword matching.

    This function scans the caption for occurrences of male- and
    female-associated terms. If both are present the first match wins.
    If neither is present the function returns ``"neutral"``. The
    keyword lists include a broad set of English and Arabic terms
    associated with male and female genders, including family roles
