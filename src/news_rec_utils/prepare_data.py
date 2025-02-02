import pandas as pd
from typing import Iterable, Optional
from .config import NewsDataset, EMBEDDING_SYSTEM_PROMPT
from pathlib import Path
import argparse

# import click

# TO_DO add logging


def read_data(
    data_dir: Path,
    news_dataset: NewsDataset,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    behaviors = pd.read_csv(
        data_dir / "raw" / news_dataset.value / "behaviors.tsv",
        sep="\t",
        header=None,
        names=["ImpressionID", "UserID", "Time", "History", "Impressions"],
        parse_dates=["Time"],
    )
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
    news["news_text"] = news.apply(
        lambda x: f"Title: {x['Title']}\nAbstract: {x['Abstract']}\nCategory: {x['Category']}\nSubCategory: {x['SubCategory']}",
        axis=1,
    )

    return behaviors, news


def process_news(news_df: pd.DataFrame) -> pd.DataFrame:
    news_df["news_text"] = news_df.apply(
        lambda x: f"Title: {x['Title']}\nAbstract: {x['Abstract']}\nCategory: {x['Category']}\nSubCategory: {x['SubCategory']}",
        axis=1,
    )
    return news_df


def filter_news(news_df: pd.DataFrame, impressions: Iterable[str]) -> pd.DataFrame:
    return news_df[news_df["NewsID"].isin(impressions)]


def remove_no_history_users(behaviors_df: pd.DataFrame) -> pd.DataFrame:

    no_history_users = behaviors_df[behaviors_df["History"].isna()]["UserID"].unique()
    return behaviors_df[~behaviors_df["UserID"].isin(no_history_users)].reset_index(
        drop=True
    )


def split_behaviors(behaviors_df: pd.DataFrame) -> pd.DataFrame:
    behaviors_df = behaviors_df.copy(deep=True)
    behaviors_df["Split_impressions"] = behaviors_df["Impressions"].str.split()

    behaviors_split = behaviors_df.explode("Split_impressions", ignore_index=True)
    behaviors_split["impression_num"] = behaviors_split.groupby(
        "ImpressionID"
    ).cumcount()
    behaviors_split[["target_impression", "target_result"]] = behaviors_split[
        "Split_impressions"
    ].str.split("-", n=1, expand=True)
    behaviors_split["target_result"] = pd.to_numeric(behaviors_split["target_result"])
    return behaviors_split.drop(columns=["Split_impressions"])


def get_unique_behavior_text(
    behaviors_df: pd.DataFrame,
    news_df: pd.DataFrame,
    embedding_system_prompt: Optional[str] = EMBEDDING_SYSTEM_PROMPT,
) -> pd.DataFrame:
    unique_history = pd.DataFrame(
        behaviors_df["History"].dropna().unique(), columns=["History"]
    )
    unique_history = unique_history.reset_index(names="HistoryID")
    unique_history["split_history"] = (
        unique_history["History"].str.split().apply(lambda x: x[::-1])
    )
    unique_history = unique_history.explode("split_history", ignore_index=True)
    unique_history["history_num"] = unique_history.groupby("HistoryID").cumcount() + 1
    unique_history = unique_history.rename(columns={"split_history": "NewsID"})
    unique_history["NewsID"] = unique_history["NewsID"].str.strip()
    unique_history = unique_history.merge(news_df, on="NewsID")
    unique_history["text"] = unique_history.apply(
        lambda x: f"{x['history_num']}. {x['news_text']}", axis=1
    )
    history_text = (
        unique_history.groupby(["HistoryID", "History"])["text"]
        .agg(
            lambda x: "\n".join(x),
        )
        .reset_index()
    )
    if embedding_system_prompt:
        history_text["text"] = history_text["text"].apply(
            lambda x: embedding_system_prompt + x
        )
    return history_text


def get_data(
    data_dir: Path,
    news_dataset: NewsDataset,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    behaviors, news = read_data(data_dir, news_dataset)
    target_impressions = (
        behaviors["Impressions"].str.split().explode().str.slice(stop=-2).unique()
    )
    # behaviors = remove_no_history_users(behaviors)
    # behaviors_split = split_behaviors(behaviors)

    news = process_news(news)

    # history_text = get_unique_behavior_text(behaviors_split, news)
    # history_text = get_unique_behavior_text(
    #     behaviors_split[behaviors_split["History"].notna()], news
    # )
    history_text = get_unique_behavior_text(
        behaviors[behaviors["History"].notna()], news
    )
    # news = filter_news(news, target_impressions)
    news = news[news["NewsID"].isin(target_impressions)].reset_index(drop=True)
    # news = filter_news(news, behaviors_split["target_impression"])
    # return behaviors_split, news, history_text
    return behaviors, news, history_text


def store_processed_data(data_dir: Path, news_dataset: NewsDataset) -> None:
    behaviors, news_text, history_text = get_data(data_dir, news_dataset)
    (data_dir / "processed" / news_dataset.value).mkdir(parents=True, exist_ok=True)
    behaviors.to_parquet(
        data_dir / "processed" / news_dataset.value / "behaviors.parquet"
    )
    news_text.to_parquet(
        data_dir / "processed" / news_dataset.value / "news_text.parquet"
    )
    history_text.to_parquet(
        data_dir / "processed" / news_dataset.value / "history_text.parquet"
    )


# @click.command()
# @click.argument("data_dir", type=click.Path(True, dir_okay=True, path_type=Path))
# @click.argument("news_dataset", type=click.Choice(choices=NewsDataset._member_names_))
# def main(data_dir, news_dataset):
#     store_processed_data(data_dir, NewsDataset[news_dataset])


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
