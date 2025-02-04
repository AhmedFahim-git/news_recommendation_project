from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from functools import partial
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import logging
from news_rec_utils.config import (
    NewsDataset,
    DEVICE,
    HISTORY_TEXT_MAXLEN,
    NEWS_TEXT_MAXLEN,
    MODEL_PATH,
)
from news_rec_utils.batch_size_finder import get_text_train_batch_size
from news_rec_utils.modelling import get_model_and_tokenizer, output_pool
from news_rec_utils.prepare_data import split_behaviors
import pandas as pd
from torch.profiler import profile, ProfilerActivity, schedule
from contextlib import nullcontext
import os
from typing import Optional
from pathlib import Path
import numpy as np
import argparse

torch.manual_seed(1234)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="mytrain.log", level=logging.INFO, format="%(asctime)s %(message)s"
)

# to do only training params in float32


def collate_fn(input, tokenizer, history_max_len, news_text_max_len):
    history_text, news_text, labels = zip(*input)
    history_counts = pd.Series(history_text).value_counts(sort=False)
    return (
        tokenizer(
            history_counts.index.to_list(),
            max_length=history_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ),
        tokenizer(
            news_text,
            max_length=news_text_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ),
        torch.tensor(history_counts.values, dtype=torch.int32),
        torch.tensor(labels) * 2 - 1,
    )


class TextDataset(Dataset):
    def __init__(self, data_dir: Path, news_dataset: NewsDataset, batch_size: int):
        behaviors = pd.read_parquet(
            data_dir / "processed" / news_dataset.value / "behaviors.parquet"
        )
        history_text = pd.read_parquet(
            data_dir / "processed" / news_dataset.value / "history_text.parquet"
        )
        news_text = pd.read_parquet(
            data_dir / "processed" / news_dataset.value / "news_text.parquet"
        )

        behaviors = behaviors[behaviors["History"].notna()].reset_index(drop=True)
        news_text = news_text.rename(columns={"NewsID": "target_impression"})

        behaviors = behaviors.merge(history_text[["History", "text"]], on="History")

        behaviors_split = split_behaviors(
            behaviors[["ImpressionID", "Impressions", "text"]]
        )
        behaviors_split = (
            behaviors_split.merge(
                news_text[["target_impression", "news_text"]],
                on="target_impression",
            )[["ImpressionID", "text", "news_text", "target_result"]]
            .rename(columns={"target_result": "label"})
            .reset_index(drop=True)
        )
        self.num_batches = int(
            min(
                -(len(behaviors_split) // -batch_size)
                - int((len(behaviors_split) % batch_size) == 1),
                behaviors_split["label"].sum(),
            )
        )
        self.dataset_size = min(self.num_batches * batch_size, len(behaviors_split))
        self.impression_label = behaviors_split[["ImpressionID", "label"]]

        self.behaviors_split = behaviors_split.drop(columns=["ImpressionID"])
        print(
            f"Input dataset size: {len(behaviors_split)}, Using size: {self.dataset_size}"
        )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        row = self.behaviors_split.iloc[idx]
        return row["text"], row["news_text"], row["label"]


class TrainSampler(Sampler):
    def __init__(self, dataset: TextDataset, batch_size: int):
        self.impression_label = dataset.impression_label
        self.dataset_size = dataset.dataset_size
        self.num_batches = dataset.num_batches
        self.batch_size = batch_size
        self.rng = np.random.default_rng(1234)
        self.num_zero = self.dataset_size - self.impression_label["label"].sum()
        self.max_zero = self._get_max_zero()

    def _get_max_zero(self):

        zero_imp_counts = self.impression_label[self.impression_label["label"] == 0][
            "ImpressionID"
        ].value_counts()
        l = self.num_zero // len(zero_imp_counts)
        h = zero_imp_counts.max()
        while h - l > 1:
            m = l + (h - l) // 2
            if zero_imp_counts.apply(lambda x: min(x, m)).sum() >= self.num_zero:
                h = m
            else:
                l = m
        return h

    def __len__(self):
        return self.dataset_size

    def _get_slice_indices(self):
        temp_copy = self.impression_label.copy()
        mappings = pd.Series(
            data=self.rng.permutation(temp_copy["ImpressionID"].unique()),
            index=temp_copy["ImpressionID"].unique(),
        ).to_dict()
        temp_copy["ImpressionID"] = temp_copy["ImpressionID"].map(mappings)
        temp_copy = temp_copy.sort_values("ImpressionID").reset_index()
        temp_copy = temp_copy.groupby("ImpressionID").sample(
            frac=1, random_state=self.rng
        )

        temp_zero = temp_copy[temp_copy["label"] == 0][
            temp_copy[temp_copy["label"] == 0]
            .groupby("ImpressionID", sort=False)
            .cumcount()
            < self.max_zero
        ]
        temp_one = temp_copy[temp_copy["label"] == 1]

        grouped = temp_zero.groupby("ImpressionID", sort=False).cumcount() == (
            self.max_zero - 1
        )
        temp_zero = temp_zero[
            (((grouped).cumsum() > (len(temp_zero) - self.num_zero)) | (~grouped))
        ]

        if (len(temp_zero) % self.num_batches) == 0:
            zero_chunks = np.split(temp_zero["index"].values, self.num_batches)
        else:
            zero_chunks = np.split(
                temp_zero["index"].values[: -(len(temp_zero) % (self.num_batches - 1))],
                self.num_batches - 1,
            ) + [
                temp_zero["index"].values[-(len(temp_zero) % (self.num_batches - 1)) :]
            ]

        if (len(temp_one) % self.num_batches) == 0:
            one_chunks = np.split(temp_one["index"].values, self.num_batches)
        else:
            one_chunks = np.split(
                temp_one["index"].values[: -(len(temp_one) % (self.num_batches - 1))],
                self.num_batches - 1,
            ) + [temp_one["index"].values[-(len(temp_one) % (self.num_batches - 1)) :]]

        all_chunks = []
        for i in range(self.num_batches):
            all_chunks.append(zero_chunks[i])
            all_chunks.append(one_chunks[i])

        temp_copy = pd.DataFrame({"final_index": np.concatenate(all_chunks)})

        batches = np.arange(self.num_batches)
        self.rng.shuffle(batches[:-1])
        temp_copy["Final_order"] = np.repeat(batches, self.batch_size)[: len(temp_copy)]
        temp_copy = temp_copy.sort_values("Final_order")
        return temp_copy["final_index"]

    def __iter__(self):
        yield from self._get_slice_indices()


class TextModelTrainer:
    def __init__(
        self,
        data_dir: Path,
        news_dataset: NewsDataset,
        model_path,
        history_max_len=HISTORY_TEXT_MAXLEN,
        news_text_max_len=NEWS_TEXT_MAXLEN,
        device=DEVICE,
        ckpt_steps=1000,
        warmup_steps=10,
    ):
        self.device = device
        self.ckpt_steps = ckpt_steps
        self.model, self.tokenizer = get_model_and_tokenizer(
            model_path, peft_model=True, device=device
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.CosineEmbeddingLoss()
        self.pool_fn = output_pool(self.model)

        batch_size = get_text_train_batch_size(
            self.model, self.optimizer, history_max_len, news_text_max_len
        )
        print(f"Batch size: {batch_size}")

        partial_collate = partial(
            collate_fn,
            tokenizer=self.tokenizer,
            history_max_len=history_max_len,
            news_text_max_len=news_text_max_len,
        )

        train_dataset = TextDataset(data_dir, news_dataset, batch_size)
        sampler = TrainSampler(train_dataset, batch_size)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=2,
            sampler=sampler,
            collate_fn=partial_collate,
        )

        self.scaler = None
        self.cast_dtype = torch.float32

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda epoch: (
                float(epoch) / float(max(1.0, warmup_steps))
                if epoch < warmup_steps
                else 1.0
            ),
        )

        if device.type == "cuda" and torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability(device)
            if compute_capability >= (8, 0):
                self.cast_dtype = torch.bfloat16
            elif compute_capability >= (6, 0):
                self.cast_dtype = torch.float16
                self.scaler = torch.amp.GradScaler(device.type)

    def train_one_batch(self, inputs):
        history_text_tokenized, news_text_tokenized, history_repeats, label = inputs
        self.optimizer.zero_grad()
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.cast_dtype,
        ):
            text_embed = self.pool_fn(
                self.model(**history_text_tokenized.to(self.device)).last_hidden_state,
                history_text_tokenized["attention_mask"],
            )

            news_text_embed = self.pool_fn(
                self.model(**news_text_tokenized.to(self.device)).last_hidden_state,
                history_text_tokenized["attention_mask"],
            )

            loss = self.loss_fn(
                torch.repeat_interleave(
                    text_embed,
                    history_repeats.to("cuda"),
                    dim=0,
                    output_size=sum(history_repeats),
                ),
                news_text_embed,
                label.to(self.device),
            )

        if self.scaler:
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def train_one_epoch(
        self,
        epoch,
        do_batch_profiling: bool = False,
        ckpt_dir: Optional[Path] = None,
    ):

        running_loss = 0
        if do_batch_profiling:
            profile_schedule = schedule(
                skip_first=5, wait=2, warmup=2, active=4, repeat=2
            )
        last_loss = 0
        with (
            profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                schedule=profile_schedule,
            )
            if do_batch_profiling
            else nullcontext()
        ) as prof:
            for i, inputs in enumerate(tqdm(self.train_dataloader)):

                running_loss += self.train_one_batch(inputs)
                if do_batch_profiling:
                    prof.step()
                if (i + 1) % self.ckpt_steps == 0:
                    last_loss = running_loss / self.ckpt_steps  # loss per batch
                    # print("Epoch {}  batch {} loss: {}".format(epoch, i + 1, last_loss))
                    logger.info(f"Epoch {epoch+1}  batch {i+1} loss: {last_loss}")
                    print(f"Epoch {epoch+1}  batch {i+1} loss: {last_loss}")
                    # logger.info(torch.cuda.memory_summary())
                    running_loss = 0.0
                    if ckpt_dir:
                        os.makedirs(ckpt_dir, exist_ok=True)
                        self.model.save_pretrained(
                            str(ckpt_dir / f"Epoch_{epoch+1}_batch_{i+1}")
                        )
                    # break

        if do_batch_profiling:
            logger.info("GPU/GPU stats:")
            logger.info(prof.key_averages().table())
            prof.export_chrome_trace("run_trace.json")
            prof.export_memory_timeline("mem_trace.html", device="cuda:0")

    def train(
        self,
        num_epochs,
        ckpt_dir: Optional[Path] = None,
        do_batch_profiling: bool = False,
    ):
        for i in range(num_epochs):
            self.model.train()
            self.train_one_epoch(
                i, do_batch_profiling=do_batch_profiling, ckpt_dir=ckpt_dir
            )
            self.model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training setup args")
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
        "--ckpt_dir",
        type=Path,
        default=None,
        help="Select the directory for saving model checkpoints",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=50,
        help="Select the number of warmup steps",
    )

    args = parser.parse_args()

    # Ensure the data_dir is a valid directory
    if not args.data_dir.is_dir():
        parser.error(f"The path '{args.data_dir}' is not a valid directory.")

    # Convert dataset name to Enum
    news_dataset = NewsDataset[args.news_dataset]

    trainer = TextModelTrainer(
        args.data_dir, news_dataset, args.model_path, warmup_steps=args.warmup_steps
    )
    # ckpt_dir = Path("model_ckpt")
    trainer.train(1, args.ckpt_dir, False)
