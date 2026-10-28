import signal
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

class _ItemTimeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _ItemTimeout

class MathVerifyScorer:
    def __init__(self):
        self._verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )

    def compute_score(
        self,
        model_output: str,
        ground_truth_unboxed: str,
        timeout_score: float,
        per_item_timeout_s: int,
    ) -> float:
        # math_metric expects boxed GT
        gt_boxed = f"\\boxed{{{ground_truth_unboxed}}}"

        # Zero disables our outer deadline, matching the ER verifier wrapper.
        # MathVerify keeps its own internal parsing/equivalence timeouts.
        if per_item_timeout_s > 0:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(per_item_timeout_s)
        try:
            score, _ = self._verify_func([gt_boxed], [model_output])
            return float(score)
        except _ItemTimeout:
            return float(timeout_score)
        except TimeoutException:
            return float(timeout_score)
        except Exception:
            return 0.0
        finally:
            if per_item_timeout_s > 0:
                signal.alarm(0)
