import torch.nn as nn
import argparse
import yaml
import time
import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from model import Linear_regression
from loss import Loss
from data_gen import Linear_Data


import tempfile
from torch.utils.tensorboard import SummaryWriter
import mlflow



parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml", help="Training configuration file")

config = yaml.load(open(parser.parse_args().config, "r"))
exp_config = config["experiment"]
training_config = config["training"]

exp_name = exp_config["name"]

with mlflow.start_run() as run: 

    mlflow.log_param("exp_name", exp_name)

    for i in training_config:
        mlflow.log_param(i, training_config[i])

    op_dir = tempfile.mkdtemp()
    writer = SummaryWriter(log_dir=op_dir)

    m = Linear_regression()
    criteria = Loss(training_config["l2_weight"], training_config["l1_weight"])
    _optim = optim.SGD(m.parameters(), lr=training_config["lr"], momentum=training_config["momentum"])

    train_dataset = Linear_Data()
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    val_dataset = Linear_Data()
    val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=True)
    for i in range(training_config["epochs"]):
        m.train()
        train_loss = 0
        for idx, data in enumerate(train_dataloader):
            _optim.zero_grad()

            ip = data["input"]
            target = data["output"]

            op = m(ip)

            loss = criteria(op, target)
            loss.backward()
            _optim.step()

            train_loss += loss.item()

        train_loss = train_loss / (idx + 1)
        print(f"TRAINING::: Epoch {i}, step {idx}, Loss: {train_loss}")
        mlflow.log_metric("train_loss", train_loss, i)
        writer.add_scalar("train_loss", train_loss, i)

        m.eval()
        val_loss = 0
        for idx, data in enumerate(val_dataloader):

            ip = data["input"]
            target = data["output"]

            op = m(ip)

            loss = criteria(op, target)

            val_loss += loss.item()
        
        val_loss = val_loss / (idx + 1)
        print(f"VALIDATION::: Epoch {i}, step {idx}, Loss: {val_loss}")
        mlflow.log_metric("val_loss", val_loss, i)
        writer.add_scalar("val_loss", val_loss, i)

    print("Training Finished. Saving the model checkpoint")
    mlflow.pytorch.log_model(m, "model")
    mlflow.log_artifact(op_dir, "events")

    writer.close()





