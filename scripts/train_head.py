from pathlib import Path
import pandas as pd
from news_rec_utils.config import (
    NewsDataset,
    NEWS_TEXT_MAXLEN,
    HEAD_MAX_BATCH_SIZE,
    DEVICE,
)
from news_rec_utils.prepare_data import split_behaviors
from news_rec_utils.modelling import (
    get_head_model,
    get_model_and_tokenizer,
    get_text_embed_eval,
    use_head_model_eval,
    pad_and_batch,
    unpad,
)
from news_rec_utils.evaluate import score
from news_rec_utils.batch_size_finder import get_text_inference_batch_size
import torch
import numpy as np
import os
from torch import optim
from tqdm import tqdm
from typing import Optional

torch.manual_seed(1234)


def get_news_embeddings(
    data_dir: Path, news_dataset: NewsDataset, text_model, tokenizer, news_text_max_len
):
    behaviors = pd.read_parquet(
        data_dir / "processed" / news_dataset.value / "behaviors.parquet"
    )
    news_text = pd.read_parquet(
        data_dir / "processed" / news_dataset.value / "news_text.parquet"
    )

    behaviors = behaviors[behaviors["History"].isna()].reset_index(drop=True)
    news_text = news_text.rename(columns={"NewsID": "target_impression"})

    behaviors_split = split_behaviors(behaviors[["ImpressionID", "Impressions"]])

    news_text = news_text[
        news_text["target_impression"].isin(behaviors_split["target_impression"])
    ]

    news_text["news_embedding"] = get_text_embed_eval(
        text_model,
        tokenizer,
        news_text["news_text"],
        get_text_inference_batch_size(text_model, news_text_max_len),
        news_text_max_len,
    )

    behaviors_split = behaviors_split.merge(
        news_text[["target_impression", "news_embedding"]], on="target_impression"
    ).reset_index(drop=True)
    return behaviors_split


def head_preprocess_df(behaviors_split_df):
    impressions = torch.tensor(
        behaviors_split_df["ImpressionID"].values, dtype=torch.int32
    )
    embeddings = torch.tensor(
        np.stack(behaviors_split_df["news_embedding"]), dtype=torch.float32
    )
    labels = torch.tensor(behaviors_split_df["target_result"].values, dtype=torch.int32)
    return impressions, embeddings, labels


def head_train_epoch(
    model, optimizer, loss_fn, impressions, embeddings, max_batch_size, labels
):
    device = model.device
    optimizer.zero_grad()
    cur_loss, num_items = 0, 0
    for embeds, attn_mask, labs in pad_and_batch(
        impressions, embeddings, max_batch_size, labels
    ):
        result = model(
            embeddings=embeds.to(device),
            attention_mask=attn_mask.to(device=device, dtype=torch.float32),
        ).last_hidden_state
        loss = loss_fn(
            unpad(result, attn_mask).squeeze(), labs.to(device, dtype=torch.float32)
        )
        cur_loss += loss.item() * len(labs)
        num_items += len(labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad()
    return cur_loss / num_items


def head_eval_epoch(model, loss_fn, impressions, embeddings, max_batch_size, labels):
    device = model.device
    res = use_head_model_eval(model, impressions, embeddings, max_batch_size).squeeze()
    loss = loss_fn(res.to(device), labels.to(device, dtype=torch.float32)).item()

    pred_df = pd.DataFrame(
        {
            "ImpressionID": impressions.numpy(),
            "pred_scores": res.numpy(),
            "label": labels.numpy(),
        }
    )
    pred_df["preds"] = pred_df.groupby("ImpressionID")["pred_scores"].rank(
        method="min", ascending=False
    )

    scoring_df = pred_df.groupby("ImpressionID")[["preds", "label"]].agg(list)

    scores_dict = score(scoring_df)
    scores_dict["loss"] = loss

    return scores_dict


def train_head(
    data_dir: Path,
    train_news_dataset: NewsDataset,
    val_news_dataset: NewsDataset,
    epochs: int,
    head_model_path: str,
    text_embed_model_path: str,
    ckpt_dir: Optional[Path] = None,
    news_text_max_len: int = NEWS_TEXT_MAXLEN,
    head_max_batch_size=HEAD_MAX_BATCH_SIZE,
    device=DEVICE,
):

    text_model, tokenizer = get_model_and_tokenizer(
        text_embed_model_path, device=device
    )

    head_model = get_head_model(head_model_path, device=device)

    if os.path.exists("temp_train.parquet"):
        train_behaviors_split = pd.read_parquet("temp_train.parquet")
    else:
        train_behaviors_split = get_news_embeddings(
            data_dir, train_news_dataset, text_model, tokenizer, news_text_max_len
        )
        train_behaviors_split.to_parquet("temp_train.parquet", index=False)

    if os.path.exists("val_train.parquet"):
        val_behaviors_split = pd.read_parquet("val_train.parquet")
    else:
        val_behaviors_split = get_news_embeddings(
            data_dir, val_news_dataset, text_model, tokenizer, news_text_max_len
        )
        val_behaviors_split.to_parquet("val_train.parquet", index=False)

    train_impressions, train_embeddings, train_labels = head_preprocess_df(
        train_behaviors_split
    )
    val_impressions, val_embeddings, val_labels = head_preprocess_df(
        val_behaviors_split
    )

    optimizer = optim.AdamW(head_model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)

    best_val_metric = -float("inf")
    for i in tqdm(range(epochs)):
        head_model.train()
        train_epoch_loss = head_train_epoch(
            head_model,
            optimizer,
            loss_fn,
            train_impressions,
            train_embeddings,
            head_max_batch_size,
            train_labels,
        )
        head_model.eval()
        val_epoch_result = head_eval_epoch(
            head_model,
            loss_fn,
            val_impressions,
            val_embeddings,
            head_max_batch_size,
            val_labels,
        )
        val_metric = np.mean(list(val_epoch_result.values()))
        scheduler.step(val_metric)
        print(
            f"Epoch: {i}, train_loss: {train_epoch_loss}, val_loss: {val_epoch_result}"
        )
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
            head_model.save_pretrained(ckpt_dir / f"epoch_{i}")
            if val_metric > best_val_metric:
                head_model.save_pretrained(ckpt_dir / f"best_model")
        best_val_metric = max(val_metric, best_val_metric)


if __name__ == "__main__":
    train_head(
        Path("data"),
        NewsDataset.MINDsmall_train,
        NewsDataset.MINDsmall_dev,
        10,
        "configs/no_history_config.json",
        text_embed_model_path="Alibaba-NLP/gte-base-en-v1.5",
        ckpt_dir=Path("models/head_model/"),
    )
