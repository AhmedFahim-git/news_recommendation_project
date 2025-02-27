from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from functools import partial
from torch import optim
from tqdm import tqdm
import logging
from news_rec_utils.config import (
    NewsDataset,
    DEVICE,
    HISTORY_TEXT_MAXLEN,
    NEWS_TEXT_MAXLEN,
    MODEL_PATH,
    DataSubset,
)
from news_rec_utils.batch_size_finder import get_text_train_batch_size
from news_rec_utils.modelling import get_model_and_tokenizer, output_pool
from news_rec_utils.prepare_data import TextTrainDataset, load_dataset
from news_rec_utils.evaluate import evaluate_df
import pandas as pd
from torch.profiler import profile, ProfilerActivity, schedule
from contextlib import nullcontext
import os
from typing import Optional
from pathlib import Path
import numpy as np
import argparse
import json

torch.manual_seed(1234)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="mytrain.log", level=logging.INFO, format="%(asctime)s %(message)s"
)


# to do only training params in float32
def eval_model_subset(
    data_dir: Path,
    news_dataset: NewsDataset,
    num_samples: int,
    random_state: np.random.Generator,
    model,
    tokenizer,
    epoch: int,
    batch: int,
):
    behaviors, news_text_dict = load_dataset(
        data_dir, news_dataset, num_samples, DataSubset.WITH_HISTORY, random_state
    )
    score_dict = evaluate_df(
        behaviors, news_text_dict, output_dir=None, model=model, tokenizer=tokenizer
    )
    final_dict = {"epoch": epoch, "batch": batch, "scores": score_dict}
    with open("./eval_score.jsonl", "a") as f:
        f.write(json.dumps(final_dict) + "\n")


class TextModelTrainer:
    def __init__(
        self,
        data_dir: Path,
        train_news_dataset: NewsDataset,
        val_news_dataset: NewsDataset,
        num_val: int,
        model_path,
        history_max_len=HISTORY_TEXT_MAXLEN,
        news_text_max_len=NEWS_TEXT_MAXLEN,
        device=DEVICE,
        ckpt_steps=50,
        warmup_steps=100,
    ):
        self.device = device
        self.rng = np.random.default_rng(1234)
        self.ckpt_steps = ckpt_steps
        self.data_dir = data_dir
        self.val_news_dataset = val_news_dataset
        self.num_val = num_val
        self.model, self.tokenizer = get_model_and_tokenizer(
            model_path, peft_model=True, device=device
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
        self.loss_fn = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=0.7
        )
        self.pool_fn = output_pool(self.model)

        batch_size = get_text_train_batch_size(
            self.model, self.optimizer, history_max_len, news_text_max_len
        )
        print(f"Batch size: {batch_size}")

        partial_collate = partial(
            TextTrainDataset.collate_fn,
            tokenizer=self.tokenizer,
            history_max_len=history_max_len,
            news_text_max_len=news_text_max_len,
        )

        train_dataset = TextTrainDataset(
            data_dir, train_news_dataset, batch_size, rng=self.rng
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=2,
            shuffle=False,
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
        history_text_tokenized, history_index, news_text_tokenized, news_index = inputs
        self.model.train()
        self.optimizer.zero_grad()
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.cast_dtype,
        ):
            history_text_embed = self.pool_fn(
                self.model(**history_text_tokenized.to(self.device)).last_hidden_state,
                history_text_tokenized["attention_mask"],
            )

            news_text_embed = self.pool_fn(
                self.model(**news_text_tokenized.to(self.device)).last_hidden_state,
                news_text_tokenized["attention_mask"],
            )

            loss = self.loss_fn(
                history_text_embed[history_index.to(self.device)],
                *torch.chunk(news_text_embed[news_index.to(self.device)], 2),
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
                    with open("./train_loss.jsonl", "a") as f:
                        f.write(
                            json.dumps(
                                {"epoch": epoch + 1, "batch": i + 1, "loss": last_loss}
                            )
                            + "\n"
                        )
                    logger.info(f"Epoch {epoch+1}  batch {i+1} loss: {last_loss}")
                    print(f"Epoch {epoch+1}  batch {i+1} loss: {last_loss}")
                    # logger.info(torch.cuda.memory_summary())
                    running_loss = 0.0
                    eval_model_subset(
                        data_dir=self.data_dir,
                        news_dataset=self.val_news_dataset,
                        num_samples=self.num_val,
                        random_state=self.rng,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        epoch=epoch + 1,
                        batch=i + 1,
                    )

                if (i + 1) % self.ckpt_steps == 0:
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
        "train_news_dataset",
        choices=NewsDataset._member_names_,
        help="Select the news dataset",
    )

    parser.add_argument(
        "--val_news_dataset",
        choices=NewsDataset._member_names_,
        default="MINDsmall_dev",
        help="Select the news dataset",
    )

    parser.add_argument(
        "--num_val",
        type=int,
        default=100,
        help="Select the number of use to eval",
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
        default=100,
        help="Select the number of warmup steps",
    )
    parser.add_argument(
        "--ckpt_steps",
        type=int,
        default=50,
        help="Select the number of steps for checkpoint",
    )

    args = parser.parse_args()

    # Ensure the data_dir is a valid directory
    if not args.data_dir.is_dir():
        parser.error(f"The path '{args.data_dir}' is not a valid directory.")

    # Convert dataset name to Enum
    train_news_dataset = NewsDataset[args.train_news_dataset]
    val_news_dataset = NewsDataset[args.val_news_dataset]

    trainer = TextModelTrainer(
        args.data_dir,
        train_news_dataset,
        val_news_dataset,
        args.num_val,
        args.model_path,
        warmup_steps=args.warmup_steps,
        ckpt_steps=args.ckpt_steps,
    )
    trainer.train(1, args.ckpt_dir, False)
