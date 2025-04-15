import json
import os
import re
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_dataset(dataset_path):
    """
    주어진 경로에서 JSON 형식의 데이터를 로드합니다.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def normalize_answer(text):
    """
    텍스트에서 특수기호 제거 및 공백 정리
    """
    text = re.sub(r"[“”‘’·…【】『』〈〉《》]", "", text)
    text = re.sub(r"[^\w\s가-힣]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def loading_model(args):
    """
    Hugging Face 모델 및 토크나이저 로드 (KULLM 기준)
    """
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()
    return model, tokenizer


def calculate_em(prediction, ground_truths):
    """
    Exact Match 계산
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    normalized_prediction = normalize_answer(prediction)
    return float(any(normalize_answer(gt) == normalized_prediction for gt in ground_truths))


def calculate_f1(prediction, ground_truths):
    """
    F1 Score 계산
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    pred_tokens = normalize_answer(prediction).split()
    f1 = 0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = max(f1, 2 * precision * recall / (precision + recall))
    return f1


def calculate_precision(prediction, ground_truths):
    """
    Precision 계산
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    pred_tokens = normalize_answer(prediction).split()
    precision = 0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = max(precision, num_same / len(pred_tokens))
    return precision


def calculate_recall(prediction, ground_truths):
    """
    Recall 계산
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    pred_tokens = normalize_answer(prediction).split()
    recall = 0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        recall = max(recall, num_same / len(gt_tokens))
    return recall
