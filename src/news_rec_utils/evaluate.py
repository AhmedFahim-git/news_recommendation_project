import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cosine
from .config import (
    NewsDataset,
    HISTORY_TEXT_MAXLEN,
    NEWS_TEXT_MAXLEN,
    HEAD_MAX_BATCH_SIZE,
    MODEL_PATH,
    HEAD_MODEL_PATH,
)
from .batch_size_finder import get_text_inference_batch_size
from .modelling import (
    get_model_and_tokenizer,
    get_text_embed_eval,
    get_head_model,
    use_head_model_eval,
)
from .prepare_data import split_behaviors
from pathlib import Path
from typing import Optional
import argparse


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def score(input_df: pd.DataFrame):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    for ind, row in tqdm(input_df.iterrows(), total=len(input_df)):
        labels = row["label"]
        sub_ranks = row["preds"]

        lt_len = float(len(labels))

        y_true = np.array(labels, dtype="float32")
        y_score = []

        for rank in sub_ranks:
            score_rslt = 1.0 / rank
            if score_rslt < 0 or score_rslt > 1:
                raise ValueError(
                    "Line-{}: score_rslt should be int from 0 to {}".format(ind, lt_len)
                )
            y_score.append(score_rslt)

        auc = roc_auc_score(y_true, y_score)
        mrr = mrr_score(y_true, y_score)
        ndcg5 = ndcg_score(y_true, y_score, 5)
        ndcg10 = ndcg_score(y_true, y_score, 10)

        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)

    return {
        "auc": np.mean(aucs),
        "mrr": np.mean(mrrs),
        "ndcg5": np.mean(ndcg5s),
        "ndcg10": np.mean(ndcg10s),
    }


def load_dataset_and_models(
    data_dir: Path,
    news_dataset: NewsDataset,
    model_path: str,
    head_model_path: str,
    num_samples: Optional[int] = None,
):
    behaviors = pd.read_parquet(
        data_dir / "processed" / news_dataset.value / "behaviors.parquet"
    )
    news_text = pd.read_parquet(
        data_dir / "processed" / news_dataset.value / "news_text.parquet"
    )
    history_text = pd.read_parquet(
        data_dir / "processed" / news_dataset.value / "history_text.parquet"
    )
    model, tokenizer = get_model_and_tokenizer(model_path)

    head_model = get_head_model(head_model_path)
    if num_samples:
        behaviors = behaviors.iloc[:num_samples]
    return behaviors, news_text, history_text, model, tokenizer, head_model


def evaluate_part_with_history(behaviors, history_text, news_text):
    if len(behaviors) == 0:
        return None
    behaviors = behaviors.merge(
        history_text[["History", "history_embedding"]], on="History", how="left"
    )
    behaviors_split = split_behaviors(behaviors)

    behaviors_split = behaviors_split.merge(news_text, on="target_impression")

    behaviors_split["pred_scores"] = behaviors_split.apply(
        lambda x: 1 - cosine(x["history_embedding"], x["news_embedding"]), axis=1
    )

    behaviors_split["preds"] = behaviors_split.groupby("ImpressionID")[
        "pred_scores"
    ].rank(method="min", ascending=False)

    scoring_df = (
        behaviors_split.groupby("ImpressionID")[["preds", "target_result"]]
        .agg(list)
        .rename(columns={"target_result": "label"})
    )

    print("Eval scores for samples with history")
    print(score(scoring_df))

    return scoring_df


def evaluate_part_without_history(behaviors, news_text, head_model):
    if len(behaviors) == 0:
        return None
    behaviors_split = split_behaviors(behaviors)

    behaviors_split = behaviors_split.merge(news_text, on="target_impression")

    impressions = torch.tensor(
        behaviors_split["ImpressionID"].values, dtype=torch.int32
    )

    behaviors_split["pred_scores"] = (
        use_head_model_eval(
            head_model,
            impressions,
            torch.tensor(behaviors_split["news_embedding"]),
            HEAD_MAX_BATCH_SIZE,
        )
        .squeeze()
        .numpy()
    )

    behaviors_split["preds"] = behaviors_split.groupby("ImpressionID")[
        "pred_scores"
    ].rank(method="min", ascending=False)

    scoring_df = (
        behaviors_split.groupby("ImpressionID")[["preds", "target_result"]]
        .agg(list)
        .rename(columns={"target_result": "label"})
    )

    print("Eval scores for samples without history")
    print(score(scoring_df))

    return scoring_df


def evaluate_df(
    behaviors: pd.DataFrame,
    news_text: pd.DataFrame,
    history_text: pd.DataFrame,
    model,
    tokenizer,
    head_model,
):

    history_text = history_text[
        history_text["History"].isin(behaviors["History"])
    ].reset_index(drop=True)

    target_impressions = (
        behaviors["Impressions"].str.split().explode().str.slice(stop=-2).unique()
    )
    news_text = news_text[news_text["NewsID"].isin(target_impressions)].reset_index(
        drop=True
    )

    model.eval()
    head_model.eval()

    history_batch_size = get_text_inference_batch_size(model, HISTORY_TEXT_MAXLEN)
    news_text_batch_size = get_text_inference_batch_size(model, NEWS_TEXT_MAXLEN)

    print(f"History batch size: {history_batch_size}")
    print(f"News text batch size: {news_text_batch_size}")

    history_text["history_embedding"] = get_text_embed_eval(
        model,
        tokenizer,
        history_text["text"],
        history_batch_size,
        HISTORY_TEXT_MAXLEN,
    )
    news_text["news_embedding"] = get_text_embed_eval(
        model,
        tokenizer,
        news_text["news_text"],
        news_text_batch_size,
        NEWS_TEXT_MAXLEN,
    )

    news_text = news_text[["NewsID", "news_embedding"]].rename(
        columns={"NewsID": "target_impression"}
    )

    history_pred_scores = evaluate_part_with_history(
        behaviors[behaviors["History"].notna()], history_text, news_text
    )

    non_history_pred_scores = evaluate_part_without_history(
        behaviors[behaviors["History"].isna()], news_text, head_model
    )

    scoring_df = behaviors.merge(
        pd.concat(
            [
                df
                for df in [history_pred_scores, non_history_pred_scores]
                if isinstance(df, pd.DataFrame)
            ]
        ),
        on="ImpressionID",
        how="left",
    )

    print("Eval scores for overall")
    print(score(scoring_df))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a news dataset using specified models."
    )

    # Positional arguments
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the directory containing data",
    )

    parser.add_argument(
        "news_dataset",
        choices=NewsDataset._member_names_,
        help="Select the news dataset",
    )

    # Optional arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help=f"Path to the model file (default: {MODEL_PATH})",
    )

    parser.add_argument(
        "--head_model_path",
        type=str,
        default=HEAD_MODEL_PATH,
        help=f"Path to the head model file (default: {HEAD_MODEL_PATH})",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: None, meaning all samples)",
    )

    args = parser.parse_args()

    # Ensure the data_dir is a valid directory
    if not args.data_dir.is_dir():
        parser.error(f"The path '{args.data_dir}' is not a valid directory.")

    # Convert dataset name to Enum
    dataset_enum = NewsDataset[args.news_dataset]

    # Run evaluation
    evaluate_df(
        *load_dataset_and_models(
            args.data_dir,
            dataset_enum,
            args.model_path,
            args.head_model_path,
            args.num_samples,
        )
    )
