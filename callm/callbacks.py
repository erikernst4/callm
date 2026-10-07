import os
import csv
import json
import time
from datetime import datetime, timedelta
import torch
from lightning.pytorch.callbacks import Callback


class SaveValPredictionsCallback(Callback):
    """
    A callback that saves per-sample validation predictions (confidence and correctness)
    to a `val_predictions.csv` file at the end of the validation epoch.
    """

    def __init__(self):
        super().__init__()
        self.confidences = []
        self.correctness = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if outputs is None:
            return
        if "confidence" in outputs and "correct" in outputs:
            self.confidences.append(outputs["confidence"].detach().cpu())
            self.correctness.append(outputs["correct"].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.confidences or not self.correctness:
            return

        confidences = torch.cat(self.confidences, dim=0).float()
        correctness = torch.cat(self.correctness, dim=0).float()

        log_dir = trainer.log_dir or trainer.default_root_dir or os.getcwd()
        predictions_file = os.path.join(log_dir, "val_predictions.csv")
        try:
            with open(predictions_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["confidence", "correctness"])
                for conf, corr in zip(confidences.numpy(), correctness.numpy()):
                    writer.writerow([f"{conf:.6f}", f"{corr:.1f}"])
            print(
                f"Saved {len(confidences)} validation predictions to {predictions_file}"
            )
        except Exception as e:
            print(f"Failed to save validation predictions: {e}")

        self.confidences.clear()
        self.correctness.clear()


class TrainingTimeCallback(Callback):
    """
    A callback that tracks and persists elapsed training and validation time for every run.
    Logs epoch duration and cumulative elapsed time to the metrics logger (e.g. metrics.csv)
    and saves a training_time.json summary file in the run's log directory.
    """

    def __init__(self):
        super().__init__()
        self.fit_start_time = None
        self.epoch_start_time = None
        self.epoch_times = []
        self.total_pure_train_time = 0.0
        self.total_fit_elapsed_prior = 0.0
        self.start_timestamp = None
        self.val_start_time = None
        self.val_start_timestamp = None

    def on_fit_start(self, trainer, pl_module):
        self.fit_start_time = time.perf_counter()
        self.start_timestamp = datetime.now().isoformat()
        if not self.epoch_times:
            self.epoch_times = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        now = time.perf_counter()
        epoch_time = (
            (now - self.epoch_start_time) if self.epoch_start_time is not None else 0.0
        )
        self.total_pure_train_time += epoch_time
        self.epoch_times.append(epoch_time)
        elapsed_time = self.total_fit_elapsed_prior + (
            (now - self.fit_start_time) if self.fit_start_time is not None else 0.0
        )

        if pl_module is not None:
            pl_module.log("train_epoch_time", epoch_time, on_epoch=True, sync_dist=True)
            pl_module.log(
                "train_elapsed_time", elapsed_time, on_epoch=True, sync_dist=True
            )

    def on_fit_end(self, trainer, pl_module):
        if self.fit_start_time is None:
            return
        total_elapsed = self.total_fit_elapsed_prior + (
            time.perf_counter() - self.fit_start_time
        )
        end_timestamp = datetime.now().isoformat()

        if trainer.is_global_zero:
            # Log total elapsed time to logger(s)
            loggers = (
                trainer.loggers
                if hasattr(trainer, "loggers") and trainer.loggers
                else ([trainer.logger] if trainer.logger else [])
            )
            for lg in loggers:
                try:
                    lg.log_metrics(
                        {"total_elapsed_time": total_elapsed},
                        step=trainer.global_step,
                    )
                    if hasattr(lg, "save"):
                        lg.save()
                except Exception as e:
                    print(f"Warning: Failed to log total_elapsed_time to logger: {e}")

            # Persist summary to training_time.json
            log_dir = trainer.log_dir or trainer.default_root_dir or os.getcwd()
            time_data = {
                "total_elapsed_time_seconds": round(total_elapsed, 2),
                "total_elapsed_time_formatted": str(
                    timedelta(seconds=int(total_elapsed))
                ),
                "train_pure_time_seconds": round(self.total_pure_train_time, 2),
                "train_pure_time_formatted": str(
                    timedelta(seconds=int(self.total_pure_train_time))
                ),
                "epoch_times_seconds": [round(t, 2) for t in self.epoch_times],
                "start_time": self.start_timestamp,
                "end_time": end_timestamp,
            }
            try:
                os.makedirs(log_dir, exist_ok=True)
                time_file = os.path.join(log_dir, "training_time.json")
                with open(time_file, "w", encoding="utf-8") as f:
                    json.dump(time_data, f, indent=2)
                print(
                    f"[TrainingTime] Total elapsed time: {time_data['total_elapsed_time_formatted']} "
                    f"({time_data['total_elapsed_time_seconds']}s). Saved to {time_file}"
                )
            except Exception as e:
                print(f"Warning: Failed to save training_time.json: {e}")

    def on_validation_start(self, trainer, pl_module):
        # If running a standalone validation run (not part of training fit)
        if not trainer.training and self.fit_start_time is None:
            self.val_start_time = time.perf_counter()
            self.val_start_timestamp = datetime.now().isoformat()

    def on_validation_end(self, trainer, pl_module):
        # If running a standalone validation run (not part of training fit)
        if (
            not trainer.training
            and self.val_start_time is not None
            and self.fit_start_time is None
        ):
            val_elapsed = time.perf_counter() - self.val_start_time
            val_end_timestamp = datetime.now().isoformat()
            if trainer.is_global_zero:
                log_dir = trainer.log_dir or trainer.default_root_dir or os.getcwd()
                time_data = {
                    "validation_elapsed_time_seconds": round(val_elapsed, 2),
                    "validation_elapsed_time_formatted": str(
                        timedelta(seconds=int(val_elapsed))
                    ),
                    "start_time": self.val_start_timestamp,
                    "end_time": val_end_timestamp,
                }
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    time_file = os.path.join(log_dir, "validation_time.json")
                    with open(time_file, "w", encoding="utf-8") as f:
                        json.dump(time_data, f, indent=2)
                    print(
                        f"[RunTime] Validation elapsed time: {time_data['validation_elapsed_time_formatted']} "
                        f"({time_data['validation_elapsed_time_seconds']}s). Saved to {time_file}"
                    )
                except Exception as e:
                    print(f"Warning: Failed to save validation_time.json: {e}")

    def state_dict(self) -> dict:
        return {
            "total_pure_train_time": self.total_pure_train_time,
            "total_fit_elapsed_prior": self.total_fit_elapsed_prior
            + (
                (time.perf_counter() - self.fit_start_time)
                if self.fit_start_time is not None
                else 0.0
            ),
            "epoch_times": self.epoch_times,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.total_pure_train_time = state_dict.get("total_pure_train_time", 0.0)
        self.total_fit_elapsed_prior = state_dict.get("total_fit_elapsed_prior", 0.0)
        self.epoch_times = state_dict.get("epoch_times", [])


class LayerWeightsHistoryCallback(Callback):
    """
    Snapshots the probe's learned layer weights every ``every_n_steps``
    training steps and saves their evolution to ``layer_weights_history.json``
    in the log directory at the end of fit. Does nothing for probes without
    weighted layer pooling.
    """

    def __init__(self, every_n_steps: int = 50):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.snapshots = []

    @staticmethod
    def _layer_pooling(pl_module):
        layer_pooling = getattr(
            getattr(pl_module, "probe", None), "layer_pooling", None
        )
        return layer_pooling if hasattr(layer_pooling, "get_weights") else None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        layer_pooling = self._layer_pooling(pl_module)
        if layer_pooling is None or trainer.global_step % self.every_n_steps != 0:
            return
        weights = layer_pooling.get_weights().detach().cpu().tolist()
        self.snapshots.append({"step": trainer.global_step, "weights": weights})

    def on_fit_end(self, trainer, pl_module):
        if not self.snapshots:
            return

        weights_file = os.path.join(
            trainer.log_dir or os.getcwd(), "layer_weights_history.json"
        )
        # Include layer_idx mapping so we know which layers the weights correspond to
        layer_idx = getattr(pl_module, "layer_idx", None)
        layer_indices = (
            layer_idx
            if isinstance(layer_idx, list)
            else list(range(getattr(pl_module, "num_layers", None) or 0))
        )
        try:
            with open(weights_file, "w") as f:
                json.dump(
                    {"layer_indices": layer_indices, "snapshots": self.snapshots},
                    f,
                    indent=2,
                )
            print(f"Layer weight history saved to {weights_file}")
        except Exception as e:
            print(f"Failed to save layer weight history: {e}")
