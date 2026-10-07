from lightning.pytorch import LightningModule
import torch
import glob
import os
import shutil


class BaseLightningModule(LightningModule):
    """
    Base LightningModule for batched inference that accumulates per-sample
    outputs in ``self.outputs``.

    Handles periodically flushing outputs to disk (``{flush_prefix}_rank{r}_{i}.pt``)
    to save memory, adopting chunks from a previous run when resuming, and
    reloading everything at epoch end. Subclasses append to ``self.outputs`` in
    ``validation_step``, call ``_maybe_flush()``, and implement
    ``process_outputs()``.
    """

    flush_prefix: str = "temp_outputs"

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

    @property
    def log_dir(self) -> str:
        """The trainer's log directory, or the current directory."""
        return self.trainer.log_dir or os.getcwd()

    def _chunk_path(self, log_dir: str, idx: int) -> str:
        return os.path.join(
            log_dir, f"{self.flush_prefix}_rank{self.global_rank}_{idx}.pt"
        )

    def _flush_outputs(self):
        """Save current outputs to the next chunk file and clear them from memory."""
        if not self.outputs:
            return

        log_dir = self.log_dir
        os.makedirs(log_dir, exist_ok=True)
        filename = self._chunk_path(log_dir, len(self.flushed_output_files))

        torch.save(self.outputs, filename)
        self.flushed_output_files.append(filename)
        self.outputs = []

    def _maybe_flush(self):
        """Flush outputs if periodic flushing is enabled and enough have accumulated."""
        if (
            self.flush_outputs_every_n_steps > 0
            and len(self.outputs) >= self.flush_outputs_every_n_steps
        ):
            self._flush_outputs()

    def _find_resume_files(self, resume_from: str) -> list[str]:
        """Rank-0 chunk files of a previous run in ``resume_from``, in chunk order."""
        if not os.path.exists(resume_from):
            raise ValueError(f"Resume path {resume_from} does not exist.")
        pattern = os.path.join(resume_from, f"{self.flush_prefix}_rank0_*.pt")
        found_files = sorted(
            glob.glob(pattern),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        if found_files:
            print(f"Found {len(found_files)} temp files to resume from.")
        return found_files

    def _adopt_resume_files(
        self, files: list[str], log_dir: str | None = None, link: bool = False
    ):
        """
        Bring chunk files from a previous run into ``log_dir`` under this run's
        naming scheme, so they are reloaded at epoch end like freshly flushed
        chunks. Files are copied, or hard-linked when ``link`` is True (falling
        back to a copy if linking fails).
        """
        log_dir = log_dir or self.log_dir
        os.makedirs(log_dir, exist_ok=True)
        for i, src_path in enumerate(files):
            filename = self._chunk_path(log_dir, i)
            if os.path.abspath(src_path) != os.path.abspath(filename):
                if link:
                    try:
                        if os.path.exists(filename):
                            os.remove(filename)
                        os.link(src_path, filename)
                    except OSError:
                        shutil.copy(src_path, filename)
                else:
                    shutil.copy(src_path, filename)
            self.flushed_output_files.append(filename)

    def _reload_flushed_outputs(self):
        """
        Reloads all flushed output chunks, clearing the disk files if save_outputs is False.
        """
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

    def _gather_outputs(self) -> list:
        """Flush any remaining outputs (if flushing is in use) and reload all chunks."""
        if self.flushed_output_files or self.flush_outputs_every_n_steps > 0:
            self._flush_outputs()
        self._reload_flushed_outputs()
        return self.outputs

    def on_validation_epoch_end(self):
        outputs = self._gather_outputs()
        if outputs:
            self.process_outputs(outputs)
        self.outputs = []

    def process_outputs(self, outputs: list[dict]):
        """Post-process all outputs of the epoch (decode, judge, write results)."""
        raise NotImplementedError

    def configure_optimizers(self):
        # We are only doing inference
        return None
