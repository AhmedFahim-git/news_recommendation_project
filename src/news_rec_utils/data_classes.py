from torch.utils.data import Dataset
from .config import (
    NEWS_CLASSIFICATION_PROMPT,
    NewsDataset,
    DataSubset,
    HISTORY_TEXT_MAXLEN,
    NEWS_TEXT_MAXLEN,
)
from .prepare_data import (
    process_history,
    load_dataset,
    split_impressions,
    expand_items,
    group_items,
    split_impressions_pos_neg,
)
import pandas as pd
from .modelling import ClassificationHead, WeightedSumModel
import numpy as np
from scipy.stats import rankdata
from typing import Optional
from pathlib import Path
from abc import ABC, abstractmethod
import torch


class AbstractTextDataset(Dataset, ABC):
    def __init__(self, text_list, news_text_dict: dict[str, str]):
        self.text_list = text_list
        self.news_text_dict = news_text_dict

    def __len__(self):
        return len(self.text_list)

    @abstractmethod
    def __getitem__(self, idx): ...


class EmbeddingDataset(Dataset):
    def __init__(self, embeds):
        self.embeds = embeds

    def __len__(self):
        return len(self.embeds)

    def __getitem__(self, idx):
        return self.embeds[idx]


def eval_collate_fn(input, tokenizer, max_len: int):
    return tokenizer(
        input,
        max_length=max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )


class HistoryDataset(AbstractTextDataset):
    def __getitem__(self, idx):
        return process_history(self.news_text_dict, self.text_list[idx])


class NewsTextDataset(AbstractTextDataset):
    def __getitem__(self, idx):
        return self.news_text_dict[self.text_list[idx]]


class NewsClassificationDataset(AbstractTextDataset):
    def __getitem__(self, idx):
        return NEWS_CLASSIFICATION_PROMPT + self.news_text_dict[self.text_list[idx]]


from .modelling_helpers import (
    get_text_embeds_list,
    get_classification_model_eval,
    get_cos_sim_eval,
    get_weighted_model_eval,
)


class NewsData:
    def __init__(
        self,
        *,
        data_dir: Optional[Path] = None,
        news_dataset: Optional[NewsDataset] = None,
        data_subset: Optional[DataSubset] = None,
        num_samples: Optional[int] = None,
        random_state: np.random.Generator | int = 1234,
        behaviors: pd.DataFrame = pd.DataFrame([]),
        news_text_dict: Optional[dict[str, str]] = None,
    ):
        if not (len(behaviors) > 0 and news_text_dict):
            assert data_dir and news_dataset
            behaviors, news_text_dict = load_dataset(
                data_dir,
                news_dataset,
                data_subset=data_subset,
                num_samples=num_samples,
                random_state=random_state,
            )
        self.news_text_dict = news_text_dict
        self.impression_ids = behaviors["ImpressionID"]
        self.news_list, split_array, self.imp_counts, self.labels = split_impressions(
            behaviors["Impressions"]
        )
        self.news_rev_index = split_array[0]
        self.has_history = behaviors["History"].notna().any()
        self.has_without_history = behaviors["History"].isna().any()
        self.history_bool = behaviors["History"].notna()
        self.history_bool_extended = behaviors["History"].notna()[split_array[1]]
        if self.has_history:
            self.history_list, self.history_rev_index = np.unique(
                behaviors[self.history_bool]["History"], return_inverse=True
            )
            self.news_rev_index_history = split_array[0, self.history_bool_extended]
            self.news_rev_index_no_history = split_array[0, ~self.history_bool_extended]
            self.history_embeds = torch.tensor([])
            self.cos_sim_scores = np.array([])
        self.news_embeds = torch.tensor([])
        self.news_embeds_original = torch.tensor([])
        self.baseline_scores = np.array([])
        self.pred_scores = np.array([])
        self.grouped_ranked_preds = np.array([], dtype=object)

    def get_all_embeds(
        self, *, model=None, tokenizer=None, model_path: Optional[str] = None
    ):
        assert self.has_history, "History should be available in order to use this"
        self.history_embeds, self.news_embeds, self.news_embeds_original = (
            get_text_embeds_list(
                [
                    (self.history_list, HISTORY_TEXT_MAXLEN, HistoryDataset, False),
                    (self.news_list, NEWS_TEXT_MAXLEN, NewsTextDataset, False),
                    (self.news_list, NEWS_TEXT_MAXLEN, NewsClassificationDataset, True),
                ],
                news_text_dict=self.news_text_dict,
                collate_fn=eval_collate_fn,
                model_path=model_path,
                model=model,
                tokenizer=tokenizer,
            )
        )

    def get_cos_sim_embeds(
        self, *, model=None, tokenizer=None, model_path: Optional[str] = None
    ):
        assert self.has_history, "History should be available in order to use this"
        self.history_embeds, self.news_embeds = get_text_embeds_list(
            [
                (self.history_list, HISTORY_TEXT_MAXLEN, HistoryDataset, False),
                (self.news_list, NEWS_TEXT_MAXLEN, NewsTextDataset, False),
            ],
            news_text_dict=self.news_text_dict,
            collate_fn=eval_collate_fn,
            model_path=model_path,
            model=model,
            tokenizer=tokenizer,
        )

    def get_classification_embeds(
        self, *, model=None, tokenizer=None, model_path: Optional[str] = None
    ):
        self.news_embeds_original = get_text_embeds_list(
            [(self.news_list, NEWS_TEXT_MAXLEN, NewsClassificationDataset, True)],
            news_text_dict=self.news_text_dict,
            collate_fn=eval_collate_fn,
            model_path=model_path,
            model=model,
            tokenizer=tokenizer,
        )[0]

    def get_baseline_scores(
        self,
        *,
        classification_model: Optional[ClassificationHead] = None,
        classification_model_path: Optional[Path] = None,
    ):
        assert len(self.news_embeds_original) > 0, "News Embeddings must be available"
        self.baseline_scores = expand_items(
            get_classification_model_eval(
                self.news_embeds_original,
                classification_model,
                classification_model_path,
            ),
            self.news_rev_index,
            self.imp_counts,
        )
        self.pred_scores = self.baseline_scores

    def get_cos_sim_scores(self):
        assert self.has_history, "History should be available in order to use this"
        assert len(self.news_embeds) > 0, "News Embeddings must be available"
        assert len(self.history_embeds) > 0, "History Embeddings must be available"
        self.cos_sim_scores = get_cos_sim_eval(
            self.news_embeds,
            self.history_embeds,
            self.news_rev_index_history,
            self.history_rev_index,
            self.imp_counts[self.history_bool],
        )

    def get_final_score(
        self,
        *,
        alpha: Optional[float] = None,
        model: Optional[WeightedSumModel] = None,
        model_path: Optional[Path] = None,
    ):
        assert self.has_history, "History should be available in order to use this"
        assert len(self.cos_sim_scores) > 0, "Cos Sim scores must be available"
        assert len(self.baseline_scores) > 0, "Baseline scores must be available"
        self.pred_scores[self.news_rev_index_history] = get_weighted_model_eval(
            self.cos_sim_scores,
            self.baseline_scores[self.news_rev_index_history],
            alpha=alpha,
            model=model,
            model_path=model_path,
        )

    def rank_group_preds(self):
        assert len(self.pred_scores) > 0, "Pred scores must be available"
        self.grouped_ranked_preds = group_items(
            self.pred_scores,
            self.imp_counts,
            func=lambda x: rankdata(-x, method="dense"),
        )

    def get_scores_dict(self, data_subset: DataSubset = DataSubset.ALL):
        from .evaluate import score

        assert (
            len(self.grouped_ranked_preds) > 0
        ), "Grouped Ranked Pred scores must be present"
        assert len(self.labels), "Labels must be available"

        if data_subset == DataSubset.WITH_HISTORY:
            assert self.has_history, "History should be available in order to use this"
            return score(
                self.grouped_ranked_preds[self.history_bool],
                self.labels[self.history_bool],
            )
        elif data_subset == DataSubset.WITHOUT_HISTORY:
            assert (
                self.has_without_history
            ), "Without history samples should be available in order to use this"
            return score(
                self.grouped_ranked_preds[~self.history_bool],
                self.labels[~self.history_bool],
            )
        else:
            return score(self.grouped_ranked_preds, self.labels)

    def save_preds(self, output_dir: Path):
        assert (
            len(self.grouped_ranked_preds) > 0
        ), "Grouped Ranked Pred scores must be present"

        output_dir.mkdir(exist_ok=True)
        lines = [
            f"{imp} [{','.join(map(str, self.grouped_ranked_preds[i].tolist()))}]\n"
            for i, imp in enumerate(self.impression_ids)
        ]
        with open(output_dir / "predictions.txt", "w") as f:
            f.writelines(lines)


class ClassificationTrainDataset(Dataset):
    def __init__(
        self,
        news_embeds: torch.Tensor,
        news_rev_index: np.ndarray,
        imp_counts: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
    ):
        assert len(news_embeds) > 0, "We need the news embeddings for this dataset"
        self.news_embeds = news_embeds
        self.news_rev_index = news_rev_index
        self.imp_counts = imp_counts
        self.labels = labels
        self.rng = rng
        self.reset()

    def __len__(self):
        return len(self.pos_neg_indices)

    def __getitem__(self, idx):
        return (
            self.news_embeds[self.pos_neg_indices[idx, 0]],
            self.news_embeds[self.pos_neg_indices[idx, 1]],
        )

    def reset(self):
        pos_neg_indices = split_impressions_pos_neg(
            self.rng,
            grouped_news_rev_index=group_items(self.news_rev_index, self.imp_counts),
            labels=self.labels,
        )
        self.pos_neg_indices = pos_neg_indices[[0, 1]].T


def text_train_collate_fn(input, tokenizer, news_text_max_len, history_max_len):
    history_text, news_text_pos, news_text_neg, pos_baseline, neg_baseline = zip(*input)
    history_unique, history_rev_index = np.unique(history_text, return_inverse=True)
    all_news = np.concatenate((news_text_pos, news_text_neg))
    news_unique, news_rev_index = np.unique(all_news, return_inverse=True)
    return (
        tokenizer(
            history_unique.tolist(),
            max_length=history_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ),
        torch.tensor(history_rev_index, dtype=torch.int32),
        tokenizer(
            news_unique.tolist(),
            max_length=news_text_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ),
        torch.tensor(news_rev_index, dtype=torch.int32),
        torch.tensor(np.concatenate((pos_baseline, neg_baseline)), dtype=torch.float32),
    )


class TextTrainDataset(Dataset):
    def __init__(
        self,
        history_list: np.ndarray,
        history_rev_index: np.ndarray,
        news_text_dict: dict[str, str],
        news_list: np.ndarray,
        news_rev_index: np.ndarray,
        baseline_scores: np.ndarray,
        imp_counts: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
    ):
        assert len(baseline_scores > 0), "We need Baseline Scores for this Dataset"
        self.history_list = history_list
        self.history_rev_index = history_rev_index
        self.news_text_dict = news_text_dict
        self.news_list = news_list
        self.news_rev_index = news_rev_index
        self.imp_counts = imp_counts
        self.baseline_scores = baseline_scores
        self.labels = labels
        self.batch_size = batch_size
        self.rng = rng
        self.reset()

    def __len__(self):
        return len(self.pos_neg_indices)

    def __getitem__(self, idx):
        return (
            process_history(
                news_text_dict=self.news_text_dict,
                history=str(
                    self.history_list[
                        self.history_rev_index[self.pos_neg_indices[idx, 2]]
                    ]
                ),
            ),
            self.news_text_dict[str(self.news_list[self.pos_neg_indices[idx, 0]])],
            self.news_text_dict[str(self.news_list[self.pos_neg_indices[idx, 1]])],
            self.baseline_scores[self.pos_neg_indices[idx, 0]],
            self.baseline_scores[self.pos_neg_indices[idx, 1]],
        )

    def reset(self):
        permuted_index = self.rng.permutation(len(self.labels))
        pos_neg_indices = split_impressions_pos_neg(
            self.rng,
            grouped_news_rev_index=group_items(self.news_rev_index, self.imp_counts)[
                permuted_index
            ],
            labels=self.labels[permuted_index],
        )
        pos_neg_indices[2] = permuted_index[pos_neg_indices[2]]
        num_batches = -(pos_neg_indices.shape[1] // -self.batch_size)
        permuted_list = self.rng.permutation(num_batches - 1).tolist() + [
            num_batches - 1
        ]
        final_index = np.concatenate(
            [
                np.arange(i * self.batch_size, (i + 1) * self.batch_size)
                for i in permuted_list
            ]
        )[: pos_neg_indices.shape[1]]
        pos_neg_indices = pos_neg_indices[:, final_index]
        self.pos_neg_indices = pos_neg_indices.T
