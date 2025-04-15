from collections import Counter
from utils import normalize_answer

def f1_score(prediction: str, ground_truth: str):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    # 빈 입력 방지
    if not pred_tokens or not gt_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    # 공통 토큰 계산
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return {"f1": f1, "precision": precision, "recall": recall}

def exact_match_score(prediction: str, ground_truth: str):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate(question, pred, answer, is_2wiki=False):
    """
    pred: 생성된 응답
    answer: 정답 (참고 답변)
    is_2wiki: 외부 평가 기준 여부 (현재는 사용 안함)
    """
    f1_metric = f1_score(pred, answer)
    em_metric = exact_match_score(pred, answer)

    return {
        "em": em_metric,
        "f1": f1_metric["f1"],
        "precision": f1_metric["precision"],
        "recall": f1_metric["recall"],
    }
