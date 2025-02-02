import torch
import gc
from functools import partial
from .modelling import dummy_head_inputs_outputs, dummy_text_inputs_outputs

# Keys are ModelConfig_str(optimizer_type)_task_max_len. Value is batch size
BATCH_SIZES = dict()


def dummy_inference_func(model, dummy_func, max_len, batch_size):
    with torch.no_grad():
        model(
            **dummy_func(max_len=max_len, batch_size=batch_size, device=model.device)[
                "inputs"
            ]
        )


def dummy_train_func(model, optimizer, dummy_func, max_len, batch_size):
    optimizer.zero_grad()
    # with torch.autocast("cuda", torch.float16):
    model(
        **dummy_func(max_len=max_len, batch_size=batch_size, device=model.device)[
            "inputs"
        ]
    ).last_hidden_state.min().backward()
    optimizer.zero_grad()


def dummy_text_train_func(
    model, optimizer, dummy_func, history_max_len, news_text_max_len, batch_size
):
    optimizer.zero_grad()
    with torch.autocast("cuda", torch.float16):
        (
            model(
                **dummy_func(
                    max_len=history_max_len, batch_size=batch_size, device=model.device
                )["inputs"]
            ).last_hidden_state.min()
            + model(
                **dummy_func(
                    max_len=news_text_max_len,
                    batch_size=batch_size,
                    device=model.device,
                )["inputs"]
            ).last_hidden_state.min()
        ).backward()
    optimizer.zero_grad()


def check_batch_size(test_func, batch_size):
    success, error = False, False
    try:
        test_func(batch_size=batch_size)
        success = True
    except torch.cuda.OutOfMemoryError as e:
        success = False
    except Exception as e:
        error = True
        print(e)
        print("Return None for batch size")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return success, error


# To do add counter
def get_batch_size(test_func):
    low = 0
    high = 1
    not_even_one = True
    while True:
        success, error = check_batch_size(test_func, high)
        if error:
            return None

        if success:
            low, high = high, high * 2
            if low == 1:
                not_even_one = False
        elif not_even_one:
            print(
                "Even batch size 1 fits into memory, try lower max len. Returning None"
            )
            return None
        else:
            break
    while high - low > 1:
        mid = low + (high - low) // 2
        success, error = check_batch_size(test_func, mid)
        if error:
            return None
        if success:
            low = mid
        else:
            high = mid
    return low


def get_text_train_batch_size(model, optimizer, history_max_len, news_text_max_len):
    if model.device.type != "cuda":
        print("Model is on CPU. Not CUDA. Returning batch size of 5")
        return 5
    model_part = str(model.config if hasattr(model, "config") else model)
    optimizer_part = str(type(optimizer))
    task_type = "TEXT_TRAINING"
    max_len_key = f"{history_max_len}_{news_text_max_len}"
    key = model_part + optimizer_part + task_type + max_len_key
    if key not in BATCH_SIZES:
        BATCH_SIZES[key] = get_batch_size(
            partial(
                dummy_text_train_func,
                model,
                optimizer,
                dummy_text_inputs_outputs,
                history_max_len,
                news_text_max_len,
            )
        )
    return BATCH_SIZES[key]


def get_text_inference_batch_size(model, max_len):
    if model.device.type != "cuda":
        print("Model is on CPU. Not CUDA. Returning batch size of 5")
        return 5
    model_part = str(model.config if hasattr(model, "config") else model)
    task_type = "TEXT_INFERENCE"
    key = model_part + task_type + str(max_len)
    if key not in BATCH_SIZES:
        BATCH_SIZES[key] = get_batch_size(
            partial(dummy_inference_func, model, dummy_text_inputs_outputs, max_len)
        )
    return BATCH_SIZES[key]


# def get_head_train_batch_size(model, optimizer, max_len):
#     model_part = str(model.config if hasattr(model, "config") else model)
#     optimizer_part = str(type(optimizer))
#     task_type = "HEAD_TRAINING"
#     key = model_part + optimizer_part + task_type + str(max_len)
#     if key not in BATCH_SIZES:
#         BATCH_SIZES[key] = (
#             get_batch_size(
#                 partial(
#                     dummy_train_func,
#                     model,
#                     optimizer,
#                     dummy_head_inputs_outputs,
#                     max_len,
#                 )
#             )
#             * max_len
#         )
#     return BATCH_SIZES[key]


# def get_head_inference_batch_size(model, max_len):
#     model_part = str(model.config if hasattr(model, "config") else model)
#     task_type = "HEAD_INFERENCE"
#     key = model_part + task_type + str(max_len)
#     if key not in BATCH_SIZES:
#         BATCH_SIZES[key] = (
#             get_batch_size(
#                 partial(dummy_inference_func, model, dummy_head_inputs_outputs, max_len)
#             )
#             * max_len
#         )
#     return BATCH_SIZES[key]
