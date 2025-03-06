import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .config import TORCH_DTYPE, NUM_WORKERS, DEVICE, CLASSIFICATION_MODEL_BATCH_SIZE
from .data_classes import AbstractTextDataset, EmbeddingDataset
from .batch_size_finder import get_text_inference_batch_size
from .modelling import (
    get_classification_head,
    ClassificationHead,
    get_model_and_tokenizer,
    output_pool,
    WeightedSumModel,
    get_weighted_sum_model,
)
from pathlib import Path
from typing import Optional
import numpy as np
from functools import partial
from tqdm import tqdm
from typing import Type, Optional
from contextlib import nullcontext


def get_embed_from_model(
    model,
    tokenizer,
    text_list: np.ndarray,
    news_text_dict: dict[str, str],
    text_maxlen: int,
    dataset_class: Type[AbstractTextDataset],
    collate_fn,
    for_classification_head: bool = False,
):
    text_batch_size = get_text_inference_batch_size(model, text_maxlen)
    print(f"Batch size for text of {text_maxlen}: {text_batch_size}")
    text_dataset = dataset_class(text_list, news_text_dict)
    text_collate_fn = partial(collate_fn, tokenizer=tokenizer, max_len=text_maxlen)
    text_dataloader = DataLoader(
        text_dataset,
        batch_size=text_batch_size,
        collate_fn=text_collate_fn,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    if for_classification_head and hasattr(model, "peft_config"):
        model.disable_adapters()
        embeds = get_text_embed_eval(model, text_dataloader)
        model.enable_adapters()
        return embeds
    return get_text_embed_eval(model, text_dataloader)


def get_text_embed_eval(model, input_dataloader: DataLoader):
    pool_fn = output_pool(model)
    text_embed_list = []
    cast_context = (
        torch.autocast(device_type="cuda", dtype=TORCH_DTYPE)
        if DEVICE.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), cast_context:
        for inputs in tqdm(input_dataloader, desc="Embedding Text"):
            text_embed_list.append(
                pool_fn(
                    model(**inputs.to(DEVICE)).last_hidden_state,
                    inputs["attention_mask"],
                )
                .detach()
                .cpu()
            )
    return torch.concatenate(text_embed_list)


def get_text_embeds_list(
    embed_text_list: list[tuple[np.ndarray, int, Type[AbstractTextDataset], bool]],
    news_text_dict: dict[str, str],
    collate_fn,
    model_path: Optional[str] = None,
    model=None,
    tokenizer=None,
):
    assert ((model is not None) and (tokenizer is not None)) or (
        model_path is not None
    ), "Either the model and tokenizer or the model path must be provided"

    if ((model is None) or (tokenizer is None)) and (model_path is not None):
        model, tokenizer = get_model_and_tokenizer(model_path)

    assert isinstance(model, torch.nn.Module)
    model.eval()
    return [
        get_embed_from_model(
            model,
            tokenizer,
            item[0],
            news_text_dict,
            item[1],
            item[2],
            collate_fn,
            item[3],
        )
        for item in embed_text_list
    ]


def get_classification_model_eval(
    embeds: torch.Tensor,
    model: Optional[ClassificationHead] = None,
    model_path: Optional[Path] = None,
):
    if not model:
        model = get_classification_head(model_path)
    embed_dataset = EmbeddingDataset(embeds)
    embed_dataloader = DataLoader(
        embed_dataset, batch_size=CLASSIFICATION_MODEL_BATCH_SIZE, shuffle=False
    )

    result_list = []
    with torch.no_grad():
        for embed in embed_dataloader:
            result_list.append(model(embed.to(DEVICE)).detach().cpu().squeeze().numpy())
    return np.concatenate(result_list)


def get_weighted_model_eval(
    cos_sim: np.ndarray | torch.Tensor,
    baseline: np.ndarray | torch.Tensor,
    model: Optional[WeightedSumModel] = None,
    model_path: Optional[Path] = None,
):
    if not model:
        model = get_weighted_sum_model(model_path)
    with torch.no_grad():
        res = (
            model(
                torch.tensor(cos_sim, device=DEVICE),
                torch.tensor(baseline, device=DEVICE),
            )
            .detach()
            .cpu()
            .numpy()
        )
    return res


def get_cos_sim_eval(
    news_embeds: torch.Tensor,
    history_embeds: torch.Tensor,
    news_rev_index: np.ndarray,
    history_rev_index: np.ndarray,
    imp_counts: np.ndarray,
):
    cos_sim = []

    cumsum_lengths = np.concatenate([[0], imp_counts.cumsum()])

    history_embeds = history_embeds[torch.tensor(history_rev_index, dtype=torch.int32)]

    for i in tqdm(range(len(imp_counts)), desc="Finding similarity"):
        cos_sim.append(
            F.cosine_similarity(
                history_embeds[[i]].to(DEVICE),
                news_embeds[
                    torch.tensor(
                        news_rev_index[cumsum_lengths[i] : cumsum_lengths[i + 1]],
                        dtype=torch.int32,
                    )
                ].to(DEVICE),
            )
            .detach()
            .cpu()
            .numpy()
        )
    return np.concatenate(cos_sim)
