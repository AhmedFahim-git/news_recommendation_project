import argparse
from news_rec_utils.config import NewsDataset, MODEL_PATH
from news_rec_utils.trainer import ClassificationModelTrainer, TextModelTrainer
from pathlib import Path


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
        "--classification_ckpt_dir",
        type=Path,
        default=None,
        help="Select the directory for saving Classification model checkpoints",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        default=None,
        help="Select the directory for saving model checkpoints",
    )
    parser.add_argument(
        "--weight_ckpt_dir",
        type=Path,
        default=None,
        help="Select the directory for saving model checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Select the directory for saving logs",
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

    classification_trainer = ClassificationModelTrainer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        train_dataset=train_news_dataset,
        val_dataset=val_news_dataset,
        ckpt_dir=args.classification_ckpt_dir,
        log_dir=args.log_dir,
    )
    classification_trainer.train(20)

    trainer = TextModelTrainer(
        model_path=args.model_path,
        train_news_data=classification_trainer.train_news_data,
        val_news_data=classification_trainer.val_news_data,
        classification_model=classification_trainer.model,
        warmup_steps=args.warmup_steps,
        ckpt_steps=args.ckpt_steps,
        log_dir=args.log_dir,
    )
    trainer.train(1, args.ckpt_dir, args.weight_ckpt_dir, False)
