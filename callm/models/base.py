from lightning.pytorch import LightningModule
import torch
import os


class BaseLightningModule(LightningModule):
    """
    Base LightningModule that handles flushing outputs to disk
    to save memory during batched inference, as well as reloading them.
    """

    def __init__(
        self,
        flush_outputs_every_n_steps: int = -1,
        save_outputs: bool = False,
    ):
        super().__init__()
        self.flush_outputs_every_n_steps = flush_outputs_every_n_steps
        self.save_outputs = save_outputs

        self.outputs = []
        self.flushed_output_files = []

    def _flush_outputs(self, prefix: str = "temp_outputs"):
        """Helper to save current outputs to a temporary file."""
        if not self.outputs:
            return

        # Use trainer log_dir or current directory
        log_dir = self.trainer.log_dir or os.getcwd()
        os.makedirs(log_dir, exist_ok=True)

        batch_idx = len(self.flushed_output_files)
        filename = os.path.join(
            log_dir, f"{prefix}_rank{self.global_rank}_{batch_idx}.pt"
        )

        torch.save(self.outputs, filename)
        self.flushed_output_files.append(filename)
        self.outputs = []  # Clear memory

    def _reload_flushed_outputs(self):
        """
        Reloads all flushed output chunks, clearing the disk files if save_outputs is False.
        """
        if self.outputs and (
            self.flushed_output_files or self.flush_outputs_every_n_steps > 0
        ):
            # Pass a default prefix assuming it won't be mixed. In subclasses we might
            # override `_flush_outputs` or pass the prefix explicitly if needed, but here
            # we just call it with default. We'll let the children classes explicitly call
            # self._flush_outputs(prefix) before reloading if they want a specific prefix.
            # However, normally the child will just call _reload_flushed_outputs.
            pass

        all_outputs = []
        for filepath in self.flushed_output_files:
            try:
                chunk = torch.load(filepath, map_location="cpu", weights_only=False)
                all_outputs.extend(chunk)
            except Exception as e:
                print(f"Error loading flushed file {filepath}: {e}")
            finally:
                # Clean up file
                if not self.save_outputs and os.path.exists(filepath):
                    os.remove(filepath)

        self.flushed_output_files = []
        self.outputs = all_outputs + self.outputs

    def configure_optimizers(self):
        # We are only doing inference
        return None
