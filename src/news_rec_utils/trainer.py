import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity, schedule
import numpy as np
import pandas as pd
from .config import (
    NewsDataset,
    DataSubset,
    NEWS_TEXT_MAXLEN,
    HISTORY_TEXT_MAXLEN,
    CLASSIFICATION_MODEL_BATCH_SIZE,
    EMBEDDING_DIM,
    DEVICE,
)
from .modelling import (
    ClassificationHead,
    WeightedSumModel,
    get_model_and_tokenizer,
    output_pool,
)
from .data_classes import (
    NewsData,
    ClassificationTrainDataset,
    TextTrainDataset,
    text_train_collate_fn,
)
from .batch_size_finder import get_text_train_batch_size
from functools import partial
from pathlib import Path
from typing import Optional
from contextlib import nullcontext
import json
from tqdm import tqdm
from datetime import datetime


class ClassificationModelTrainer:
    def __init__(
        self,
        model_path: str,
        data_dir: Path,
        train_dataset: NewsDataset,
        val_dataset: NewsDataset,
        train_subset: DataSubset = DataSubset.ALL,
        val_subset: DataSubset = DataSubset.ALL,
        log_dir: Optional[Path] = None,
        news_text_maxlen: int = NEWS_TEXT_MAXLEN,
        ckpt_dir: Optional[Path] = None,
    ):
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.rng = np.random.default_rng(1234)
        self.train_news_data = NewsData(
            data_dir=data_dir,
            news_dataset=train_dataset,
            data_subset=train_subset,
            random_state=self.rng,
        )
        self.val_news_data = NewsData(
            data_dir=data_dir,
            news_dataset=val_dataset,
            data_subset=val_subset,
            random_state=self.rng,
        )

        self.train_news_data.get_classification_embeds(model_path=model_path)
        self.val_news_data.get_classification_embeds(model_path=model_path)
        self.train_dataset = ClassificationTrainDataset(
            self.train_news_data.news_embeds_original,
            self.train_news_data.news_rev_index,
            self.train_news_data.imp_counts,
            self.train_news_data.labels,
            self.rng,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=CLASSIFICATION_MODEL_BATCH_SIZE, shuffle=True
        )

        self.model = ClassificationHead(
            in_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, out_dim=1
        ).to(device=DEVICE)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MarginRankingLoss(2)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=2
        )

    def train_one_epoch(self, epoch: int):
        self.model.train()
        losses, counts = [], []
        for pos_embeds, neg_embeds in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            pos_res = self.model(pos_embeds.to(device=DEVICE)).squeeze()
            neg_res = self.model(neg_embeds.to(device=DEVICE)).squeeze()
            loss = self.loss_fn(
                pos_res, neg_res, torch.tensor([1], dtype=torch.int32, device=DEVICE)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            losses.append(loss.item())
            counts.append(len(pos_embeds))
        self.optimizer.zero_grad()
        train_epoch_loss = np.dot(losses, counts) / sum(counts)

        self.train_news_data.get_baseline_scores(classification_model=self.model)
        self.val_news_data.get_baseline_scores(classification_model=self.model)

        self.train_news_data.rank_group_preds()
        self.val_news_data.rank_group_preds()

        train_eval_score = self.train_news_data.get_scores_dict()
        val_eval_score = self.val_news_data.get_scores_dict()

        print(
            epoch + 1,
            train_epoch_loss,
            "\nTrain Score:",
            train_eval_score,
            "\nVal Score:",
            val_eval_score,
        )
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir / "train_classification_score.jsonl", "a") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "epoch": epoch + 1,
                            "scores": train_eval_score,
                            "loss": train_epoch_loss,
                        }
                    )
                    + "\n"
                )
            with open(self.log_dir / "eval_classification_score.jsonl", "a") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "epoch": epoch + 1,
                            "scores": val_eval_score,
                        }
                    )
                    + "\n"
                )
        self.scheduler.step(np.mean(list(val_eval_score.values())))
        if self.ckpt_dir:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.ckpt_dir / f"Epoch_{epoch+1}.pt")
            pd.DataFrame(
                {
                    "NewsID": self.train_news_data.news_list[
                        self.train_news_data.news_rev_index
                    ],
                    "Preds": self.train_news_data.expand_baseline_scores(),
                }
            ).to_csv(self.ckpt_dir / f"Epoch_{epoch+1}_train.csv", index=False)
            pd.DataFrame(
                {
                    "NewsID": self.val_news_data.news_list[
                        self.val_news_data.news_rev_index
                    ],
                    "Preds": self.val_news_data.expand_baseline_scores(),
                }
            ).to_csv(self.ckpt_dir / f"Epoch_{epoch+1}_val.csv", index=False)

    def train(self, num_epochs: int):
        for i in range(num_epochs):
            self.train_one_epoch(i)
            self.train_dataset.reset()


class TextModelTrainer:
    def __init__(
        self,
        model_path: str,
        data_dir: Optional[Path] = None,
        train_news_dataset: Optional[NewsDataset] = None,
        val_news_dataset: Optional[NewsDataset] = None,
        num_val: Optional[int] = None,
        train_news_data: Optional[NewsData] = None,
        val_news_data: Optional[NewsData] = None,
        classification_model=None,
        classification_model_path: Optional[Path] = None,
        history_max_len=HISTORY_TEXT_MAXLEN,
        news_text_max_len=NEWS_TEXT_MAXLEN,
        log_dir: Optional[Path] = None,
        ckpt_dir: Optional[Path] = None,
        weight_ckpt_dir: Optional[Path] = None,
        device=DEVICE,
        ckpt_steps=50,
        warmup_steps=100,
    ):
        self.device = device
        self.rng = np.random.default_rng(1234)
        self.ckpt_steps = ckpt_steps
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.weight_ckpt_dir = weight_ckpt_dir
        if train_news_data:
            self.train_news_data = train_news_data
        else:
            assert data_dir and train_news_dataset
            self.train_news_data = NewsData(
                data_dir=data_dir,
                news_dataset=train_news_dataset,
                data_subset=DataSubset.WITH_HISTORY,
            )
        if val_news_data:
            self.val_news_data = val_news_data
        else:
            assert data_dir and val_news_dataset
            self.val_news_data = NewsData(
                data_dir=data_dir,
                news_dataset=val_news_dataset,
                data_subset=DataSubset.WITH_HISTORY,
                num_samples=num_val,
            )
        self.model, self.tokenizer = get_model_and_tokenizer(
            model_path, peft_model=True, device=device
        )
        self.weight_model = WeightedSumModel()
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.weight_model.parameters()),
            lr=1e-5,
        )
        if len(self.train_news_data.baseline_scores) == 0:
            if len(self.train_news_data.news_embeds_original) == 0:
                self.train_news_data.get_classification_embeds(
                    model=self.model, tokenizer=self.tokenizer
                )
            self.train_news_data.get_baseline_scores(
                classification_model=classification_model,
                classification_model_path=classification_model_path,
            )
        if len(self.val_news_data.baseline_scores) == 0:
            if len(self.val_news_data.news_embeds_original) == 0:
                self.val_news_data.get_classification_embeds(
                    model=self.model, tokenizer=self.tokenizer
                )
            self.val_news_data.get_baseline_scores(
                classification_model=classification_model,
                classification_model_path=classification_model_path,
            )

        self.loss_fn = torch.nn.MarginRankingLoss(2)
        self.pool_fn = output_pool(self.model)

        batch_size = get_text_train_batch_size(
            self.model, self.optimizer, history_max_len, news_text_max_len
        )
        print(f"Batch size: {batch_size}")

        partial_collate = partial(
            text_train_collate_fn,
            tokenizer=self.tokenizer,
            history_max_len=history_max_len,
            news_text_max_len=news_text_max_len,
        )

        train_dataset = TextTrainDataset(
            self.train_news_data.history_list,
            self.train_news_data.history_rev_index,
            self.train_news_data.news_text_dict,
            self.train_news_data.news_list,
            self.train_news_data.news_rev_index_history,
            self.train_news_data.expand_baseline_scores()[
                self.train_news_data.history_bool_extended
            ],
            self.train_news_data.imp_counts[self.train_news_data.history_bool],
            self.train_news_data.labels[self.train_news_data.history_bool],
            batch_size=batch_size,
            rng=self.rng,
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
        (
            history_text_tokenized,
            history_index,
            news_text_tokenized,
            news_index,
            pos_neg_baseline,
        ) = inputs
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

            pos_neg_cos_sim = F.cosine_similarity(
                history_text_embed[history_index.repeat(2).to(self.device)],
                news_text_embed[news_index.to(self.device)],
            )

            loss = self.loss_fn(
                *torch.chunk(
                    self.weight_model(
                        pos_neg_baseline.to(self.device), pos_neg_cos_sim
                    ),
                    2,
                ),
                torch.tensor([1], device=self.device, dtype=torch.float32),
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
                    assert prof is not None
                    prof.step()
                if (i + 1) % self.ckpt_steps == 0:
                    last_loss = running_loss / self.ckpt_steps  # loss per batch
                    if self.log_dir:
                        self.log_dir.mkdir(parents=True, exist_ok=True)
                        with open(self.log_dir / "train_loss.jsonl", "a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "timestamp": datetime.now().isoformat(),
                                        "epoch": epoch + 1,
                                        "batch": i + 1,
                                        "loss": last_loss,
                                    }
                                )
                                + "\n"
                            )
                    print(f"Epoch {epoch+1}  batch {i+1} loss: {last_loss}")
                    running_loss = 0.0
                    self.val_news_data.get_cos_sim_embeds(
                        model=self.model, tokenizer=self.tokenizer
                    )
                    self.val_news_data.get_cos_sim_scores()
                    self.val_news_data.get_final_score(model=self.weight_model)
                    self.val_news_data.rank_group_preds()
                    val_score_dict = self.val_news_data.get_scores_dict(
                        DataSubset.WITH_HISTORY
                    )
                    print(val_score_dict)
                    if self.log_dir:
                        self.log_dir.mkdir(parents=True, exist_ok=True)
                        with open(self.log_dir / "eval_score.jsonl", "a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "timestamp": datetime.now().isoformat(),
                                        "epoch": epoch + 1,
                                        "batch": i + 1,
                                        "scores": val_score_dict,
                                    }
                                )
                                + "\n"
                            )

                if (i + 1) % self.ckpt_steps == 0:
                    if self.ckpt_dir:
                        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                        self.model.save_pretrained(
                            str(self.ckpt_dir / f"Epoch_{epoch+1}_batch_{i+1}")
                        )
                    if self.weight_ckpt_dir:
                        self.weight_ckpt_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            self.weight_model.state_dict(),
                            self.weight_ckpt_dir / f"Epoch_{epoch+1}_batch_{i+1}.pt",
                        )
                    # break

        if do_batch_profiling:
            assert prof is not None
            prof.export_chrome_trace("run_trace.json")
            prof.export_memory_timeline("mem_trace.html", device="cuda:0")

    def train(
        self,
        num_epochs,
        do_batch_profiling: bool = False,
    ):
        for i in range(num_epochs):
            self.model.train()
            self.train_one_epoch(
                i,
                do_batch_profiling=do_batch_profiling,
            )
            self.model.eval()
