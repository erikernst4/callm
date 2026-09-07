import os
import json
import time
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping

from callm.callbacks import TrainingTimeCallback
from cli import CalibrationTrainer


class DummyLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)

    def training_step(self, batch, batch_idx):
        loss = self.layer(batch).sum()
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.layer(batch).sum()
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_training_time_callback_fit(tmp_path):
    logger = CSVLogger(save_dir=str(tmp_path), name="test_run")
    callback = TrainingTimeCallback()
    trainer = pl.Trainer(
        max_epochs=2,
        logger=logger,
        callbacks=[callback],
        accelerator="cpu",
        log_every_n_steps=1,
    )

    dataset = torch.randn(10, 2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    model = DummyLightningModel()
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=val_loader)

    # Check metrics.csv
    metrics_path = os.path.join(logger.log_dir, "metrics.csv")
    assert os.path.exists(metrics_path), f"metrics.csv missing at {metrics_path}"
    df = pd.read_csv(metrics_path)

    assert "train_epoch_time" in df.columns
    assert "train_elapsed_time" in df.columns
    assert "total_elapsed_time" in df.columns

    # Verify values are positive numbers
    epoch_times = df["train_epoch_time"].dropna().tolist()
    assert len(epoch_times) == 2
    assert all(t > 0 for t in epoch_times)

    tot_elapsed = df["total_elapsed_time"].dropna().tolist()
    assert len(tot_elapsed) >= 1
    assert tot_elapsed[-1] > 0

    # Check training_time.json
    time_json_path = os.path.join(logger.log_dir, "training_time.json")
    assert os.path.exists(time_json_path)

    with open(time_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "total_elapsed_time_seconds" in data
    assert "total_elapsed_time_formatted" in data
    assert "train_pure_time_seconds" in data
    assert "train_pure_time_formatted" in data
    assert "epoch_times_seconds" in data
    assert len(data["epoch_times_seconds"]) == 2
    assert "start_time" in data
    assert "end_time" in data
    assert data["total_elapsed_time_seconds"] > 0


def test_training_time_callback_validation_only(tmp_path):
    logger = CSVLogger(save_dir=str(tmp_path), name="val_run")
    callback = TrainingTimeCallback()
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[callback],
        accelerator="cpu",
    )

    dataset = torch.randn(10, 2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    model = DummyLightningModel()
    trainer.validate(model, dataloaders=loader)

    val_json_path = os.path.join(logger.log_dir, "validation_time.json")
    assert os.path.exists(val_json_path)

    with open(val_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "validation_elapsed_time_seconds" in data
    assert "validation_elapsed_time_formatted" in data
    assert "start_time" in data
    assert "end_time" in data
    assert data["validation_elapsed_time_seconds"] >= 0


def test_calibration_trainer_auto_injects_training_time_callback(tmp_path):
    # Test 1: No callbacks specified
    trainer1 = CalibrationTrainer(default_root_dir=str(tmp_path), accelerator="cpu")
    cb_types1 = [type(cb) for cb in trainer1.callbacks]
    assert TrainingTimeCallback in cb_types1

    # Test 2: Existing callbacks list specified
    early_stop = EarlyStopping(monitor="val_loss")
    trainer2 = CalibrationTrainer(
        default_root_dir=str(tmp_path),
        callbacks=[early_stop],
        accelerator="cpu",
    )
    cb_types2 = [type(cb) for cb in trainer2.callbacks]
    assert EarlyStopping in cb_types2
    assert TrainingTimeCallback in cb_types2

    # Test 3: TrainingTimeCallback already in callbacks list (no duplicate)
    time_cb = TrainingTimeCallback()
    trainer3 = CalibrationTrainer(
        default_root_dir=str(tmp_path),
        callbacks=[time_cb],
        accelerator="cpu",
    )
    timing_cbs = [
        cb for cb in trainer3.callbacks if isinstance(cb, TrainingTimeCallback)
    ]
    assert len(timing_cbs) == 1


def test_training_time_callback_state_dict():
    cb = TrainingTimeCallback()
    cb.fit_start_time = time.perf_counter() - 10.0
    cb.total_pure_train_time = 8.5
    cb.epoch_times = [4.2, 4.3]

    state = cb.state_dict()
    assert state["total_pure_train_time"] == 8.5
    assert state["total_fit_elapsed_prior"] >= 10.0
    assert state["epoch_times"] == [4.2, 4.3]

    cb_new = TrainingTimeCallback()
    cb_new.load_state_dict(state)
    assert cb_new.total_pure_train_time == 8.5
    assert cb_new.total_fit_elapsed_prior == state["total_fit_elapsed_prior"]
    assert cb_new.epoch_times == [4.2, 4.3]


def test_extract_best_metrics_includes_elapsed_time(tmp_path):
    from scripts.summarize_experiment_results import extract_best_metrics

    # Scenario 1: metrics.csv has total_elapsed_time
    csv_path = tmp_path / "metrics.csv"
    df = pd.DataFrame(
        {
            "epoch": [0, 1],
            "step": [10, 20],
            "val_loss_epoch": [0.5, 0.3],
            "val_auc": [0.8, 0.9],
            "total_elapsed_time": [float("nan"), 125.4],
        }
    )
    df.to_csv(csv_path, index=False)

    metrics = extract_best_metrics(str(csv_path))
    assert "elapsed_time_sec" in metrics
    assert metrics["elapsed_time_sec"] == 125.4

    # Scenario 2: metrics.csv does not have it, but training_time.json is present
    tmp_sub = tmp_path / "sub"
    tmp_sub.mkdir()
    csv_path2 = tmp_sub / "metrics.csv"
    df2 = pd.DataFrame(
        {
            "epoch": [0, 1],
            "step": [10, 20],
            "val_loss_epoch": [0.5, 0.3],
            "val_auc": [0.8, 0.9],
        }
    )
    df2.to_csv(csv_path2, index=False)

    json_path = tmp_sub / "training_time.json"
    with open(json_path, "w") as f:
        json.dump({"total_elapsed_time_seconds": 45.6}, f)

    metrics2 = extract_best_metrics(str(csv_path2))
    assert "elapsed_time_sec" in metrics2
    assert metrics2["elapsed_time_sec"] == 45.6


def test_plot_probe_training_with_elapsed_time(tmp_path):
    from scripts.plot_probe_training import plot_metrics

    csv_path = tmp_path / "metrics.csv"
    df = pd.DataFrame(
        {
            "epoch": [0, 1],
            "step": [10, 20],
            "train_loss_step": [0.6, 0.4],
            "train_loss_epoch": [0.6, 0.4],
            "val_loss_step": [0.5, 0.3],
            "val_loss_epoch": [0.5, 0.3],
            "val_auc": [0.8, 0.9],
            "val_ece": [0.1, 0.05],
            "val_cross_entropy": [0.5, 0.3],
            "total_elapsed_time": [float("nan"), 65.0],
        }
    )
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "plots"
    plot_metrics(str(csv_path), str(out_dir))

    assert (out_dir / "probe_loss_plots.png").exists()
    assert (out_dir / "probe_loss_plots.pdf").exists()
