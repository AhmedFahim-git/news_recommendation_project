from enum import Enum
import torch
from .modified_peft import ModifiedLoraConfig
from peft.utils.peft_types import TaskType


class NewsDataset(Enum):
    MINDsmall_train = "MINDsmall_train"
    MINDsmall_dev = "MINDsmall_dev"
    MINDlarge_train = "MINDlarge_train"
    MINDlarge_dev = "MINDlarge_dev"
    MINDlarge_test = "MINDlarge_test"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "Alibaba-NLP/gte-base-en-v1.5"
HEAD_MODEL_PATH = "models/head_model/best_model"

HISTORY_TEXT_MAXLEN = 256  # Can be over 20,000
NEWS_TEXT_MAXLEN = 256  # Actually close to 600

HEAD_MAXLEN = 600  # Actual is around 600

# Max batch size should be greater than the HEAD_MAXLEN; smaller number usually
# works better due to large differences in number of samples for each impression
HEAD_MAX_BATCH_SIZE = 1024

EMBEDDING_DIM = 768


# TRAIN_TEXT_BATCH_SIZE = 10  # todo

# INFER_HISTORY_BATCH_SIZE = 32
# INFER_NEWS_BATCH_SIZE = 128


# HEAD_BATCH_SIZE = 512

EMBEDDING_SYSTEM_PROMPT = """Objective: Learn the user's news reading preferences based on past articles to predict whether they would read a given future news article.
Below are the news articles read by the user in reverse chronological order. Please analyze these articles and extract patterns in the user's reading behavior,such as their preferred categories, subcategories, topics, tone, and any other relevant features. We will use this information to predict whether the user would read a given news article.
User's Past News Articles (in reverse chronological order):
"""

TORCH_DTYPE = torch.float32

if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability(DEVICE)
    if compute_capability >= (8, 0):
        TORCH_DTYPE = torch.bfloat16
    elif compute_capability >= (6, 0):
        TORCH_DTYPE = torch.float16

LORA_CONFIG = ModifiedLoraConfig(
    r=16,
    target_modules=["qkv_proj"],
    task_type=TaskType.FEATURE_EXTRACTION,
    lora_alpha=32,
    lora_dropout=0.05,
    enable_lora=[True, False, True],
)

# EMPTY_USER_PROMPT = (
#     "Given that the user hasn't read any news yet what news to recommend"
# )
