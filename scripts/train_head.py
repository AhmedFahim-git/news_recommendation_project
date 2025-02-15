from pathlib import Path
import pandas as pd
from news_rec_utils.config import (
    NewsDataset,
    NEWS_TEXT_MAXLEN,
    HEAD_MAX_BATCH_SIZE,
    DEVICE,
    NUM_WORKERS,
    DataSubset,
)
from news_rec_utils.prepare_data import (
    split_impressions,
    eval_collate_fn,
    NewsTextDataset,
    load_dataset,
)
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
from torch.utils.data import DataLoader
import numpy as np
import os
from torch import optim
from tqdm import tqdm
from typing import Optional
from functools import partial
from scipy.stats import rankdata

torch.manual_seed(1234)


def get_news_embeddings(
    news_list,
    news_text_dict,
    news_text_max_len: int,
    model,
    tokenizer,
):

    model.eval()
    news_text_batch_size = get_text_inference_batch_size(model, news_text_max_len)

    print(f"News text batch size: {news_text_batch_size}")

    news_dataset = NewsTextDataset(news_list, news_text_dict)
    news_collate_fn = partial(
        eval_collate_fn, tokenizer=tokenizer, max_len=NEWS_TEXT_MAXLEN
    )
    news_dataloader = DataLoader(
        news_dataset,
        batch_size=news_text_batch_size,
        collate_fn=news_collate_fn,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    news_embeds = get_text_embed_eval(model, news_dataloader)
    return news_embeds


def head_train_epoch(
    model,
    optimizer,
    loss_fn,
    max_batch_size,
    news_rev_index,
    news_embeds,
    imp_counts,
    labels,
):
    device = model.device
    optimizer.zero_grad()
    cur_loss, num_items = 0, 0
    for sub_emb_index, attn_mask, labs in pad_and_batch(
        imp_counts, news_rev_index, max_batch_size, np.concatenate(labels)
    ):
        attn_mask = torch.tensor(attn_mask, dtype=torch.int32)
        result = model(
            embeddings=news_embeds[torch.tensor(sub_emb_index, dtype=torch.int32)].to(
                device
            ),
            attention_mask=attn_mask.to(device=device, dtype=torch.float32),
        ).last_hidden_state
        loss = loss_fn(
            unpad(result, attn_mask).squeeze(),
            torch.tensor(labs, device=device, dtype=torch.float32),
        )
        cur_loss += loss.item() * len(labs)
        num_items += len(labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad()
    return cur_loss / num_items


def head_eval_epoch(
    model, loss_fn, max_batch_size, news_rev_index, news_embeds, imp_counts, labels
):
    device = model.device

    result_list = []
    cumsum_lengths = np.concatenate([[0], imp_counts.cumsum()])

    model.eval()
    print("Evaluating using head")
    pred_scores = use_head_model_eval(
        model, imp_counts, news_rev_index, news_embeds, max_batch_size
    ).squeeze()

    for i in range(len(imp_counts)):
        result_list.append(
            rankdata(
                -pred_scores.numpy()[cumsum_lengths[i] : cumsum_lengths[i + 1]],
                method="dense",
            ).tolist()
        )
    scores_dict = score(np.array(result_list, dtype=object), labels)
    loss = loss_fn(
        pred_scores.to(device),
        torch.tensor(np.concatenate(labels), device=device, dtype=torch.float32),
    ).item()
    scores_dict["loss"] = loss

    return scores_dict


def get_train_val_embed(
    model_path,
    device,
    train_news_list,
    train_news_text_dict,
    val_news_list,
    val_news_text_dict,
    max_len,
):
    text_model, tokenizer = get_model_and_tokenizer(model_path, device=device)
    train_news_embed = get_news_embeddings(
        train_news_list, train_news_text_dict, max_len, text_model, tokenizer
    )
    val_news_embed = get_news_embeddings(
        val_news_list, val_news_text_dict, max_len, text_model, tokenizer
    )
    return train_news_embed, val_news_embed


def get_dataset(data_dir: Path, news_dataset: NewsDataset):
    behaviors, news_text_dict = load_dataset(
        data_dir, news_dataset, data_subset=DataSubset.WITHOUT_HISTORY
    )
    news_list, split_array, imp_counts, labels = split_impressions(
        behaviors["Impressions"]
    )
    return news_text_dict, news_list, split_array, imp_counts, labels


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

    (
        train_news_text_dict,
        train_news_list,
        train_split_array,
        train_imp_counts,
        train_labels,
    ) = get_dataset(data_dir, train_news_dataset)
    (
        val_news_text_dict,
        val_news_list,
        val_split_array,
        val_imp_counts,
        val_labels,
    ) = get_dataset(data_dir, val_news_dataset)

    train_news_embed, val_news_embed = get_train_val_embed(
        text_embed_model_path,
        device,
        train_news_list,
        train_news_text_dict,
        val_news_list,
        val_news_text_dict,
        news_text_max_len,
    )

    head_model = get_head_model(head_model_path, device=device)

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
            head_max_batch_size,
            train_split_array[0],
            train_news_embed,
            train_imp_counts,
            train_labels,
        )
        head_model.eval()
        val_epoch_result = head_eval_epoch(
            head_model,
            loss_fn,
            head_max_batch_size,
            val_split_array[0],
            val_news_embed,
            val_imp_counts,
            val_labels,
        )
        val_metric = float(np.mean(list(val_epoch_result.values())))
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
