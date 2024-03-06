import argparse
import base64
import json
import logging
import os

import torch
import wandb
import msgpack
import cloudpickle
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src import config, features
from src.data import DataModule, S3ModelLoader, lazy_frame_loader
from src.model import {{ cookiecutter.project_slug }}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

cloudpickle.register_pickle_by_value(features)


def load_state(model, model_path):
    with S3ModelLoader(model_path) as f:
        if f:
            state = torch.load(f)
            model.load_state_dict(state_dict=state["model"])
            print("Model loaded from S3")
    return model


# This is the main method, to be run when train.py is invoked
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # # Data, model, and output directories. Passed by sagemaker with default to os env variables
    parser.add_argument(
        "-tr", "--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument(
        "-o", "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument(
        "-m", "--model-dir", type=str, default=os.environ["SM_MODEL_DIR"]
    )
    parser.add_argument("-u", "--model-uri", type=str, default="")
    parser.add_argument("-r", "--runmeta", type=str, default="")
    parser.add_argument("-k", "--wandb", type=str, default="")
    parser.add_argument("-c", "--config", type=str)

    args, _ = parser.parse_known_args()
    logger.info("args: %s", args)

    decoded_msg = base64.b64decode(args.config)
    config_data = msgpack.loads(decoded_msg)
    nn_config = config.NNConfig(**config_data)

    runmeta = json.loads(base64.b64decode(args.runmeta).decode("utf-8"))
    runmeta.update(config_data["hyperparameters"])

    wandb.login(key=args.wandb)
    wandb.init(project=runmeta["model-name"], config=runmeta)
    wandb_logger = WandbLogger()

    model = {{ cookiecutter.project_slug }}(config=nn_config)

    logger.info(f"Loading pretrained model from {args.model_uri}")
    model = load_state(model, args.model_uri)

    transformer = features.FeatureEngineeringTransformer(
        feature_list=nn_config.features(),
        meta_features=["cohort_date", "dx", "cohort_size", nn_config.label_name],
        **config_data,
    )

    for i, batch in enumerate(lazy_frame_loader(args.train)):
        logger.info(f"Fitting transformer on batch {i}")
        transformer.fit(batch.to_pandas())

    data_module = DataModule(
        data_path=args.train, transformer=transformer, config=nn_config
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="/opt/ml/checkpoints",  
        filename="checkpoint",  
        save_top_k=1,
        save_last=True, 
        monitor="avg_train_loss",  
        mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor="avg_train_loss", 
        min_delta=0.0005,  
        patience=3, 
        verbose=True, 
        mode="min"
    )

    logger.info("Starting training")
    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=nn_config.hyperparameters.n_epochs,
        default_root_dir=args.output_data_dir,
        inference_mode=False,
        fast_dev_run=False,
        logger=wandb_logger,
    )

    trainer.fit(model, data_module)
    wandb.finish()

    logger.info("Training complete. Saving model")
    model.export_model("model.onnx")
    with open(os.path.join(args.model_dir, "bundle.pkl"), "wb") as f:
        cloudpickle.dump(
            {
                "transformer": transformer,
                "onnx": open("model.onnx", "rb").read(),
                "ckpt": open(f"/opt/ml/checkpoints/checkpoint.ckpt", "rb").read(),
            },
            f,
        )
