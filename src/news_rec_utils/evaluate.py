import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
from .config import (
    NewsDataset,
    DataSubset,
    HISTORY_TEXT_MAXLEN,
    NEWS_TEXT_MAXLEN,
    HEAD_MAX_BATCH_SIZE,
    MODEL_PATH,
    HEAD_MODEL_PATH,
    NUM_WORKERS,
    DEVICE,
)
from .prepare_data import (
    load_dataset,
)
from pathlib import Path
from typing import Optional
import argparse

# Scoring functions adapted from https://github.com/msnews/MIND/blob/master/evaluate.py


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


def score(preds_input, labels_input):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    for ind in tqdm(range(len(preds_input))):
        labels = labels_input[ind]
        sub_ranks = preds_input[ind]

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


# def get_history_news_text_embeds(
#     history_list,
#     news_list,
#     news_text_dict,
#     model_path: Optional[str] = None,
#     model=None,
#     tokenizer=None,
# ):
#     assert ((model is not None) and (tokenizer is not None)) or (
#         model_path is not None
#     ), "Either the model and tokenizer or the model path must be provided"

#     if ((model is None) or (tokenizer is None)) and (model_path is not None):
#         model, tokenizer = get_model_and_tokenizer(model_path)

#     assert isinstance(model, torch.nn.Module)
#     model.eval()
#     history_embeds = get_embed_from_model(
#         model,
#         tokenizer,
#         history_list,
#         news_text_dict,
#         HISTORY_TEXT_MAXLEN,
#         HistoryDataset,
#     )
#     news_embeds = get_embed_from_model(
#         model, tokenizer, news_list, news_text_dict, NEWS_TEXT_MAXLEN, NewsTextDataset
#     )
# history_batch_size = get_text_inference_batch_size(model, HISTORY_TEXT_MAXLEN)
# news_text_batch_size = get_text_inference_batch_size(model, NEWS_TEXT_MAXLEN)

# print(f"History batch size: {history_batch_size}")
# print(f"News text batch size: {news_text_batch_size}")

# history_dataset = HistoryDataset(history_list, news_text_dict)
# history_collate_fn = partial(
#     eval_collate_fn, tokenizer=tokenizer, max_len=HISTORY_TEXT_MAXLEN
# )
# history_dataloader = DataLoader(
#     history_dataset,
#     batch_size=history_batch_size,
#     collate_fn=history_collate_fn,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=NUM_WORKERS,
# )

# news_dataset = NewsTextDataset(news_list, news_text_dict)
# news_collate_fn = partial(
#     eval_collate_fn, tokenizer=tokenizer, max_len=NEWS_TEXT_MAXLEN
# )
# news_dataloader = DataLoader(
#     news_dataset,
#     batch_size=news_text_batch_size,
#     collate_fn=news_collate_fn,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=NUM_WORKERS,
# )

# history_embeds = get_text_embed_eval(model, history_dataloader)
# news_embeds = get_text_embed_eval(model, news_dataloader)
# return history_embeds, news_embeds


# def evaluate_part_with_history(
#     filtered_news_rev_index, history_embeds, news_embeds, imp_counts, history_rev_index
# ):
#     cos_sim = torch.nn.CosineSimilarity()
#     result_list = []
#     cumsum_lengths = np.concatenate([[0], imp_counts.cumsum()])

#     history_embeds = history_embeds[torch.tensor(history_rev_index, dtype=torch.int32)]

#     for i in tqdm(range(len(imp_counts)), desc="Finding similarity"):
#         result_list.append(
#             rankdata(
#                 -cos_sim(
#                     history_embeds[[i]].to(DEVICE),
#                     news_embeds[
#                         torch.tensor(
#                             filtered_news_rev_index[
#                                 cumsum_lengths[i] : cumsum_lengths[i + 1]
#                             ],
#                             dtype=torch.int32,
#                         )
#                     ].to(DEVICE),
#                 ).cpu(),
#                 method="dense",
#             ).tolist()
#         )

#     return np.array(result_list, dtype=object)


# def evaluate_part_without_history(
#     filtered_news_rev_index,
#     news_embeds,
#     imp_counts,
#     head_model_path: Optional[str] = None,
#     head_model=None,
# ):
#     if len(filtered_news_rev_index) == 0:
#         return np.array([], dtype=object)
#     assert (head_model is not None) or (
#         head_model_path is not None
#     ), "Either the head model or the head model path must be provided"

#     if head_model_path:
#         head_model = get_head_model(head_model_path)

#     assert isinstance(head_model, torch.nn.Module)
#     head_model.eval()

#     result_list = []
#     cumsum_lengths = np.concatenate([[0], imp_counts.cumsum()])

#     print("Evaluating using head")
#     pred_scores = (
#         use_head_model_eval(
#             head_model, imp_counts, filtered_news_rev_index, news_embeds
#         )
#         .squeeze()
#         .numpy()
#     )

#     for i in range(len(imp_counts)):
#         result_list.append(
#             rankdata(
#                 -pred_scores[cumsum_lengths[i] : cumsum_lengths[i + 1]], method="dense"
#             ).tolist()
#         )

#     return np.array(result_list, dtype=object)

from .data_classes import NewsData


def evaluate_df(
    behaviors: pd.DataFrame,
    news_text_dict: dict[str, str],
    output_dir: Optional[Path] = None,
    model_path: Optional[str] = None,
    model=None,
    tokenizer=None,
    classification_model_path: Optional[Path] = None,
    classification_model=None,
    alpha: Optional[float] = None,
    weight_model=None,
    weith_model_path=None,
):
    final_score_dict = dict()
    news_data_obj = NewsData(behaviors=behaviors, news_text_dict=news_text_dict)
    news_data_obj.get_all_embeds(
        model=model, tokenizer=tokenizer, model_path=model_path
    )
    news_data_obj.get_cos_sim_scores()
    news_data_obj.get_baseline_scores(
        classification_model=classification_model,
        classification_model_path=classification_model_path,
    )
    news_data_obj.get_final_score(
        alpha=alpha, model=weight_model, model_path=weith_model_path
    )
    news_data_obj.rank_group_preds()

    if len(news_data_obj.labels) > 0:
        if news_data_obj.has_history:
            final_score_dict["with_history_score"] = news_data_obj.get_scores_dict(
                DataSubset.WITH_HISTORY
            )
        if news_data_obj.has_without_history:
            final_score_dict["without_history_score"] = news_data_obj.get_scores_dict(
                DataSubset.WITHOUT_HISTORY
            )
        final_score_dict["overall_score_dict"] = news_data_obj.get_scores_dict(
            DataSubset.ALL
        )

    print(final_score_dict)

    if output_dir:
        news_data_obj.save_preds(output_dir)

    # news_list, split_array, imp_counts, labels = split_impressions(
    #     behaviors["Impressions"]
    # )

    # history_list, history_rev_index = np.unique(
    #     behaviors[behaviors["History"].notna()]["History"], return_inverse=True
    # )

    # history_embeds, news_embeds, news_embeds_original = get_text_embeds_list(
    #     [
    #         (history_list, HISTORY_TEXT_MAXLEN, HistoryDataset, False),
    #         (news_list, NEWS_TEXT_MAXLEN, NewsTextDataset, False),
    #         (news_list, NEWS_TEXT_MAXLEN, NewsClassificationDataset, True),
    #     ],
    #     news_text_dict=news_text_dict,
    #     model_path=model_path,
    #     model=model,
    #     tokenizer=tokenizer,
    # )
    # baseline_scores = get_classification_model_eval(
    #     news_embeds_original, classification_model, classification_model_path
    # )

    # history_embeds, news_embeds = get_history_news_text_embeds(
    #     history_list,
    #     news_list,
    #     news_text_dict,
    #     model_path=model_path,
    #     model=model,
    #     tokenizer=tokenizer,
    # )

    # history_result = evaluate_part_with_history(
    #     split_array[0, behaviors["History"].notna().values[split_array[1]]],
    #     history_embeds,
    #     news_embeds,
    #     imp_counts[behaviors["History"].notna()],
    #     history_rev_index,
    # )

    # without_history_result = evaluate_part_without_history(
    #     split_array[0, behaviors["History"].isna().values[split_array[1]]],
    #     news_embeds,
    #     imp_counts[behaviors["History"].isna()],
    #     head_model_path=classification_model_path,
    #     head_model=classification_model,
    # )

    # all_preds = np.empty((len(behaviors),), dtype=object)

    # all_preds[behaviors["History"].notna().values] = history_result
    # all_preds[behaviors["History"].isna().values] = without_history_result

    # if len(labels) > 0:
    #     if len(history_result) > 0:
    #         "Eval scores for samples with history"
    #         final_score_dict["with_history_score"] = score(
    #             history_result, labels[behaviors["History"].notna()]
    #         )
    #         print(final_score_dict["with_history_score"])
    #     if len(without_history_result) > 0:
    #         "Eval scores for samples without history"
    #         final_score_dict["without_history_score"] = score(
    #             without_history_result, labels[behaviors["History"].isna()]
    #         )
    #         print(final_score_dict["without_history_score"])
    #     print("Eval scores for overall")
    #     final_score_dict["overall_score_dict"] = score(all_preds, labels)
    #     print(final_score_dict["overall_score_dict"])

    # if output_dir:
    #     output_dir.mkdir(exist_ok=True)
    #     lines = [
    #         f"{imp} [{','.join(map(str, all_preds[i]))}]\n"
    #         for i, imp in enumerate(behaviors["ImpressionID"])
    #     ]
    #     with open(output_dir / "predictions.txt", "w") as f:
    #         f.writelines(lines)
    # return final_score_dict


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
        "--output_dir",
        type=Path,
        default=Path("."),
        help=f"Path to the head model file (default: {Path('.')})",
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
    behaviors, news_text = load_dataset(args.data_dir, dataset_enum, args.num_samples)

    evaluate_df(
        behaviors,
        news_text,
        output_dir=args.output_dir,
        model_path=args.model_path,
        classification_model_path=args.head_model_path,
    )
