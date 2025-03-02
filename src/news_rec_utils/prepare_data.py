import pandas as pd
from .config import (
    NewsDataset,
    EMBEDDING_SYSTEM_PROMPT,
    DataSubset,
    NEWS_CLASSIFICATION_PROMPT,
    NEWS_TEXT_MAXLEN,
    HISTORY_TEXT_MAXLEN,
)
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Optional
import numpy as np
from torch.utils.data import Dataset
import torch
from abc import ABC, abstractmethod
from typing import Callable


def split_impressions_pos_neg(
    rng: np.random.Generator, grouped_news_rev_index: np.ndarray, labels: np.ndarray
):
    pos_ind, neg_ind, len_list = [], [], []
    for i, row in enumerate(labels):
        temp_pos, temp_neg = [], []
        num_pos = sum(row)
        num_neg = len(row) - num_pos
        max_len = max(num_pos, num_neg)
        for j, label in enumerate(row):
            news_rev_ind = grouped_news_rev_index[i][j]
            if label == 0:
                temp_neg.append(news_rev_ind)
            else:
                temp_pos.append(news_rev_ind)
        temp_pos = rng.permutation(
            np.append(temp_pos, rng.choice(temp_pos, max_len - num_pos))
        )
        temp_neg = rng.permutation(
            np.append(temp_neg, rng.choice(temp_neg, max_len - num_neg))
        )

        pos_ind.extend(temp_pos.tolist())
        neg_ind.extend(temp_neg.tolist())
        len_list.append(max_len)
    return np.stack(
        [
            np.array(pos_ind, dtype=np.int32),
            np.array(neg_ind, dtype=np.int32),
            np.concatenate([[i] * n for i, n in enumerate(len_list)], dtype=np.int32),
        ]
    )


# def split_impressions_pos_neg(
#     rng: np.random.Generator, impressions
# ) -> tuple[np.ndarray, np.ndarray]:
#     cur = 0
#     pos_dict = dict()
#     len_list = []
#     news_list = []
#     pos_ind = []
#     neg_ind = []
#     for row in tqdm(impressions, desc="Splitting impressions"):
#         temp_pos, temp_neg = [], []
#         news, label = zip(
#             *map(lambda x: (x[0], int(x[1])), [k.split("-") for k in row.split()])
#         )
#         num_pos = sum(label)
#         num_neg = len(label) - num_pos
#         max_len = max(num_neg, num_pos)
#         for i in range(len(label)):
#             if news[i] not in pos_dict:
#                 pos_dict[news[i]] = cur
#                 cur += 1
#                 news_list.append(news[i])
#             if label[i] == 0:
#                 temp_neg.append(pos_dict[news[i]])
#             else:
#                 temp_pos.append(pos_dict[news[i]])
#         temp_pos = rng.permutation(
#             np.append(temp_pos, rng.choice(temp_pos, max_len - num_pos))
#         )
#         temp_neg = rng.permutation(
#             np.append(temp_neg, rng.choice(temp_neg, max_len - num_neg))
#         )

#         pos_ind.extend(temp_pos.tolist())
#         neg_ind.extend(temp_neg.tolist())
#         len_list.append(max_len)

#     return np.array(news_list), np.stack(
#         [
#             np.array(pos_ind, dtype=np.int32),
#             np.array(neg_ind, dtype=np.int32),
#             np.concatenate([[i] * n for i, n in enumerate(len_list)], dtype=np.int32),
#         ]
#     )


class AbstractTextTrainDataset(Dataset, ABC):
    def __init__(
        self,
        data_dir: Path,
        news_dataset: NewsDataset,
        batch_size: int,
        include_history: bool,
        data_subset: DataSubset,
        rng=np.random.default_rng(1234),
    ):
        self.rng = rng
        behaviors, self.news_text_dict = load_dataset(
            data_dir, news_dataset, data_subset=data_subset
        )

        behaviors = behaviors.sample(
            frac=1, replace=False, random_state=self.rng
        ).reset_index(drop=True)
        if include_history:
            self.history = behaviors["History"].values

        self.news_list, self.final_array = split_impressions_pos_neg(
            self.rng, behaviors["Impressions"]
        )

        num_batches = -(self.final_array.shape[1] // -batch_size)
        permuted_list = self.rng.permutation(num_batches - 1).tolist() + [
            num_batches - 1
        ]
        final_index = np.concatenate(
            [np.arange(i * batch_size, (i + 1) * batch_size) for i in permuted_list]
        )[: self.final_array.shape[1]]
        self.final_array = self.final_array[:, final_index]

        print(
            "Input dataset size: {}, Using size: {}".format(
                behaviors["Impressions"].str.count("N").sum(), self.final_array.shape[1]
            )
        )

    def __len__(self):
        return self.final_array.shape[1]

    @abstractmethod
    def __getitem__(self, idx): ...

    @staticmethod
    @abstractmethod
    def collate_fn(input, tokenizer, news_text_max_len, history_max_len): ...


class TextTrainDataset(AbstractTextTrainDataset):
    def __init__(
        self,
        data_dir: Path,
        news_dataset: NewsDataset,
        batch_size: int,
        baseline_dict: dict[str, float],
        rng=np.random.default_rng(1234),
    ):
        super().__init__(
            data_dir,
            news_dataset,
            batch_size,
            include_history=True,
            data_subset=DataSubset.WITH_HISTORY,
            rng=rng,
        )
        self.baseline_dict = baseline_dict

    def __getitem__(self, idx):
        return (
            process_history(
                self.news_text_dict, str(self.history[self.final_array[2, idx]])
            ),
            self.news_text_dict[str(self.news_list[self.final_array[0, idx]])],
            self.news_text_dict[str(self.news_list[self.final_array[1, idx]])],
            self.baseline_dict[str(self.news_list[self.final_array[0, idx]])],
            self.baseline_dict[str(self.news_list[self.final_array[1, idx]])],
        )

    @staticmethod
    def collate_fn(input, tokenizer, news_text_max_len, history_max_len):
        history_text, news_text_pos, news_text_neg, pos_base, neg_base = zip(*input)
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
            torch.tensor(pos_base, dtype=torch.float32),
            torch.tensor(neg_base, dtype=torch.float32),
        )


def process_history(news_text_dict: dict[str, str], history: str):
    return EMBEDDING_SYSTEM_PROMPT + "\n".join(
        [f"{i+1}. {news_text_dict[x]}" for i, x in enumerate(history.split()[::-1])]
    )


class TextTrainSequenceClassification(AbstractTextTrainDataset):
    def __init__(
        self,
        data_dir: Path,
        news_dataset: NewsDataset,
        batch_size: int,
        data_subset: DataSubset,
        rng=np.random.default_rng(1234),
    ):
        super().__init__(
            data_dir,
            news_dataset,
            batch_size,
            include_history=False,
            data_subset=data_subset,
            rng=rng,
        )

    def __getitem__(self, idx):
        return (
            NEWS_CLASSIFICATION_PROMPT
            + self.news_text_dict[str(self.news_list[self.final_array[0, idx]])],
            NEWS_CLASSIFICATION_PROMPT
            + self.news_text_dict[str(self.news_list[self.final_array[1, idx]])],
        )

    @staticmethod
    def collate_fn(input, tokenizer, news_text_max_len, history_max_len=0):
        news_text_pos, news_text_neg = zip(*input)
        all_news = np.concatenate((news_text_pos, news_text_neg))
        news_unique, news_rev_index = np.unique(all_news, return_inverse=True)
        return (
            tokenizer(
                news_unique.tolist(),
                max_length=news_text_max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ),
            torch.tensor(news_rev_index, dtype=torch.int32),
        )


def expand_items(items: np.ndarray, rev_index: np.ndarray, imp_counts: np.ndarray):
    result_list = []

    cumsum_lengths = np.concatenate([[0], imp_counts.cumsum()])
    for i in range(len(imp_counts)):
        result_list.append(items[rev_index[cumsum_lengths[i] : cumsum_lengths[i + 1]]])
    return np.concatenate(result_list)


def group_items(
    items: np.ndarray,
    imp_counts: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray] = lambda x: x,
):
    result_list = []

    cumsum_lengths = np.concatenate([[0], imp_counts.cumsum()])
    for i in range(len(imp_counts)):
        result_list.append(func(items[cumsum_lengths[i] : cumsum_lengths[i + 1]]))
    return np.array(result_list, dtype=object)


def load_dataset(
    data_dir: Path,
    news_dataset: NewsDataset,
    num_samples: Optional[int] = None,
    data_subset: Optional[DataSubset] = DataSubset.ALL,
    random_state: int | np.random.Generator = 1234,
):
    behaviors = pd.read_parquet(
        data_dir / "processed" / news_dataset.value / "behaviors.parquet",
        columns=["ImpressionID", "History", "Impressions"],
    )
    news_text_dict: dict[str, str] = (
        pd.read_parquet(
            data_dir / "processed" / news_dataset.value / "news_text.parquet"
        )
        .set_index("NewsID")["news_text"]
        .to_dict()
    )
    if data_subset == DataSubset.WITH_HISTORY:
        behaviors = behaviors[behaviors["History"].notna()].reset_index(drop=True)
    elif data_subset == DataSubset.WITHOUT_HISTORY:
        behaviors = behaviors[behaviors["History"].isna()].reset_index(drop=True)
    if num_samples and num_samples < len(behaviors):
        behaviors = behaviors.sample(
            n=num_samples, random_state=random_state, replace=False
        ).reset_index(drop=True)
        # behaviors = behaviors.iloc[:num_samples]

    return behaviors, news_text_dict


def read_data(
    data_dir: Path,
    news_dataset: NewsDataset,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Reading behaviors.tsv data")
    behaviors = pd.read_csv(
        data_dir / "raw" / news_dataset.value / "behaviors.tsv",
        sep="\t",
        header=None,
        names=["ImpressionID", "UserID", "Time", "History", "Impressions"],
        parse_dates=["Time"],
    )

    print("Reading news.tsv data")
    news = pd.read_csv(
        data_dir / "raw" / news_dataset.value / "news.tsv",
        sep="\t",
        header=None,
        names=[
            "NewsID",
            "Category",
            "SubCategory",
            "Title",
            "Abstract",
            "URL",
            "Title Entities",
            "Abstract Entities",
        ],
    )

    return behaviors, news


def process_news(news_df: pd.DataFrame) -> pd.DataFrame:
    print("Making news_text column for the news data")
    news_df["news_text"] = news_df.apply(
        lambda x: f"Title: {x['Title']}\nAbstract: {x['Abstract']}\nCategory: {x['Category']}\nSubCategory: {x['SubCategory']}",
        axis=1,
    )
    return news_df


def split_impressions(impressions):
    assert len(impressions) > 0, "No Impressions given"
    label_present = "-" in impressions[0]
    cur = 0
    pos_dict = dict()
    news_list = []
    rev_ind = []
    labels = []
    len_list = []
    for row in tqdm(impressions, desc="Splitting impressions"):
        if label_present:
            news_sub_list, label = zip(
                *map(lambda x: (x[0], int(x[1])), [k.split("-") for k in row.split()])
            )
            labels.append(label)
        else:
            news_sub_list = row.split()
        len_list.append(len(news_sub_list))
        for news in news_sub_list:
            if news not in pos_dict:
                pos_dict[news] = cur
                cur += 1
                news_list.append(news)

            rev_ind.append(pos_dict[news])
    return (
        np.array(news_list),
        np.stack(
            [
                np.array(rev_ind, dtype=np.int32),
                np.concatenate(
                    [[i] * n for i, n in enumerate(len_list)], dtype=np.int32
                ),
            ]
        ),
        np.array(len_list, dtype=np.int32),
        np.array(labels, dtype=object),
    )


def get_data(
    data_dir: Path,
    news_dataset: NewsDataset,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    behaviors, news = read_data(data_dir, news_dataset)

    print("Getting list of target impressions for final filtering of news dataset")

    news = process_news(news)
    return behaviors, news


def store_processed_data(data_dir: Path, news_dataset: NewsDataset) -> None:
    behaviors, news_text = get_data(data_dir, news_dataset)

    print("Saving datasets")
    (data_dir / "processed" / news_dataset.value).mkdir(parents=True, exist_ok=True)
    behaviors.to_parquet(
        data_dir / "processed" / news_dataset.value / "behaviors.parquet"
    )
    news_text.to_parquet(
        data_dir / "processed" / news_dataset.value / "news_text.parquet"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process news dataset and store the results."
    )

    # Argument for the data directory
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the directory containing data",
    )

    # Argument for selecting a dataset
    parser.add_argument(
        "news_dataset",
        choices=NewsDataset._member_names_,
        help="Select the news dataset",
    )

    args = parser.parse_args()

    # Ensure the data_dir is a valid directory
    if not args.data_dir.is_dir():
        parser.error(f"The path '{args.data_dir}' is not a valid directory.")

    # Convert dataset name to Enum
    dataset_enum = NewsDataset[args.news_dataset]

    # Call the processing function
    store_processed_data(args.data_dir, dataset_enum)
