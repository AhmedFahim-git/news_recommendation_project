from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
from .config import DEVICE, EMBEDDING_DIM, TORCH_DTYPE, HEAD_MAX_BATCH_SIZE, LORA_CONFIG
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPast,
)
from tqdm import tqdm
from contextlib import nullcontext
from collections.abc import Callable
import numpy as np
from peft.mapping import get_peft_model

# path = "Alibaba-NLP/gte-base-en-v1.5"


def dummy_text_inputs_outputs(batch_size: int, max_len: int, device=DEVICE):
    return {
        "inputs": {
            "input_ids": torch.ones((batch_size, max_len), dtype=torch.int).to(device),
            "attention_mask": torch.ones((batch_size, max_len), dtype=torch.int).to(
                device
            ),
            "token_type_ids": torch.zeros((batch_size, max_len), dtype=torch.int).to(
                device
            ),
        },
        "outputs": torch.ones((batch_size,), dtype=torch.float32, device=device),
    }


def dummy_head_inputs_outputs(
    batch_size: int, max_len: int, embedding_dim: int = EMBEDDING_DIM, device=DEVICE
):
    return {
        "inputs": {
            "embeddings": torch.rand((batch_size, max_len, embedding_dim)).to(device),
            "attention_mask": torch.ones((batch_size, max_len), dtype=torch.int32).to(
                device
            ),
        },
        "outputs": torch.ones(
            (batch_size * max_len,), dtype=torch.float32, device=device
        ),
    }


@torch.compile
def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def output_pool(model) -> Callable[[torch.Tensor, ...], torch.Tensor]:
    def get_last_embedding(
        last_hidden_states: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return last_hidden_states[:, 0]

    if model.config.architectures[0] == "Qwen2ForCausalLM":
        return last_token_pool
    elif model.config.architectures[0] == "NewModel":
        return get_last_embedding
    # Alternate implementation
    # with torch.no_grad():
    #     output = model(**dummy_text_inputs(1, 1, device=model.device)).to("cpu")
    # if isinstance(output, BaseModelOutputWithPast):
    #     return last_token_pool
    # elif isinstance(output, BaseModelOutputWithPooling):
    #     return get_last_embedding


# As a final task please add an attention layer on top of the news embeddings to get modified embeddings that takes all news into account
def get_model_and_tokenizer(path: str, peft_model: bool = False, device=DEVICE):
    model = AutoModel.from_pretrained(
        path,
        trust_remote_code=True,
        unpad_inputs=True,
        use_memory_efficient_attention=True,
        # torch_dtype=torch.float16,
    ).to(device)
    if peft_model and not hasattr(model, "peft_config"):
        model = get_peft_model(model, LORA_CONFIG)

    if hasattr(model, "peft_config"):
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    return model, tokenizer


# Add attention layer over embeddings
def get_sequence_classification_model(path, device=DEVICE):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        trust_remote_code=True,
        unpad_inputs=True,
        use_memory_efficient_attention=True,
        # torch_dtype=torch.float16,
        num_labels=1,
    ).to(device)

    for param in model.parameters():
        param.requires_grad = False

    for param in list(model.parameters())[-4:]:
        param.requires_grad = True

    return model, tokenizer


# To do. Try replacing with dataloader
def get_text_embed_eval(model, input_dataloader):
    device = model.device
    pool_fn = output_pool(model)
    text_embed_list = []
    cast_context = (
        torch.autocast(device_type="cuda", dtype=TORCH_DTYPE)
        if model.device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), cast_context:
        for inputs in tqdm(input_dataloader, desc="Embedding Text"):
            text_embed_list.append(
                pool_fn(
                    model(**inputs.to(device)).last_hidden_state,
                    inputs["attention_mask"],
                )
                .detach()
                .cpu()
            )
    return torch.concatenate(text_embed_list)


def get_head_model(path: str, device=DEVICE):
    if path.endswith(".json"):
        my_config = AutoConfig.from_pretrained(path, trust_remote_code=True)

        my_model = AutoModel.from_config(my_config, trust_remote_code=True)
        return my_model.to(device)
    else:
        return AutoModel.from_pretrained(path, trust_remote_code=True).to(device)


def pad_and_batch(
    imp_lengths: np.ndarray,
    embed_index: np.ndarray,
    max_batch_size: int = HEAD_MAX_BATCH_SIZE,
    labels: np.ndarray = np.array([]),
):
    labs = np.array([])
    cumsum_lengths = np.concatenate([[0], imp_lengths.cumsum()])
    assert max(imp_lengths) <= max_batch_size
    start = 0
    end = 0
    while end < len(imp_lengths):
        end = (cumsum_lengths <= (cumsum_lengths[start] + max_batch_size)).sum() - 1
        max_len = imp_lengths[start:end].max()
        attn_mask = np.stack(
            [
                np.pad(
                    np.ones(i, dtype=np.int32),
                    (0, max_len - i),
                    mode="constant",
                    constant_values=0,
                )
                for i in imp_lengths[start:end]
            ]
        )

        sub_embeds_index = np.stack(
            [
                np.pad(
                    embed_index[cumsum_lengths[i] : cumsum_lengths[i + 1]],
                    (0, max_len - imp_lengths[i]),
                    mode="constant",
                    constant_values=0,
                )
                for i in range(start, end)
            ]
        )
        if len(labels) > 0:
            labs = labels[cumsum_lengths[start] : cumsum_lengths[end]]
        yield sub_embeds_index, attn_mask, labs
        start = end


def unpad(embeds, attn_mask):
    sequence_lengths = attn_mask.sum(dim=1)
    return torch.cat([embeds[i, :l] for i, l in enumerate(sequence_lengths)])


def use_head_model_eval(
    model: torch.nn.Module,
    imp_count: np.ndarray,
    emb_index: np.ndarray,
    embeddings: torch.Tensor,
    max_batch_size: int = HEAD_MAX_BATCH_SIZE,
) -> torch.Tensor:
    device = model.device
    pred_scores = []

    for sub_emb_index, attn_mask, _ in pad_and_batch(
        imp_count, emb_index, max_batch_size
    ):
        sub_emb = embeddings[torch.tensor(sub_emb_index, dtype=torch.int32)]
        attn_mask = torch.tensor(attn_mask, dtype=torch.int32)
        with torch.no_grad():
            res = model(
                sub_emb.to(device), attn_mask.to(device=device, dtype=torch.float32)
            ).last_hidden_state
        pred_scores.append(unpad(res.cpu(), attn_mask))

    return torch.cat(pred_scores)


def use_head_model_eval_with_history(
    model: torch.nn.Module,
    imp_count: np.ndarray,
    news_rev_index: np.ndarray,
    news_embeds: torch.Tensor,
    history_embed: torch.Tensor,
    history_rev_index,
    max_batch_size: int = HEAD_MAX_BATCH_SIZE,
):
    device = model.device
    num_items = 0
    pred_scores = []
    for sub_emb_index, attn_mask, _ in pad_and_batch(
        imp_count,
        news_rev_index,
        max_batch_size,
    ):
        sub_history = history_embed[history_rev_index][
            num_items : num_items + len(attn_mask)
        ].unsqueeze(1)
        extra_attn = np.ones((len(attn_mask), 1), dtype=np.int32)
        num_items += len(attn_mask)
        attn_mask = torch.tensor(
            np.concatenate([extra_attn, attn_mask], axis=1), dtype=torch.int32
        )
        with torch.no_grad():
            result = model(
                embeddings=torch.cat(
                    [
                        sub_history,
                        news_embeds[torch.tensor(sub_emb_index, dtype=torch.int32)],
                    ],
                    dim=1,
                ).to(device),
                attention_mask=attn_mask.to(device=device, dtype=torch.float32),
            ).last_hidden_state
        pred_scores.append(unpad(result[:, 1:].detach().cpu(), attn_mask[:, 1:]))
    return torch.cat(pred_scores)
