import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from callm.models.llm import LLM

class Probe(nn.Module):
    """
    A simple probe that gets single layer activations, mean pools across sequence,
    and applies a linear layer + sigmoid to return the confidence directly.
    """
    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.linear = nn.Linear(model_dim, 1)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: shape (#batch, #seq, #model_dim)
        Returns:
            confidence: shape (#batch)
        """
        # Mean pool over the sequence dimension (dimension 1)
        # shape after mean: (#batch, #model_dim)
        pooled = activations.mean(dim=1)
        # shape after linear: (#batch, 1)
        logits = self.linear(pooled)
        # shape after squeeze: (#batch)
        return torch.sigmoid(logits.squeeze(-1))


class ConfidenceAugmentedLLM(LightningModule):
    """
    LightningModule that wraps a Probe and a LLM module.
    Runs the LLM with gradients disabled to extract activations,
    and runs the Probe with gradients enabled to train it to predict confidence.
    """
    def __init__(
        self,
        llm: LLM,
        probe: nn.Module = None,
        layer_idx: int = -1,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.llm = llm
        self.layer_idx = layer_idx
        self.lr = lr

        # Retrieve model_dim from the LLM model configuration
        if hasattr(self.llm.model.config, "hidden_size"):
            self.model_dim = self.llm.model.config.hidden_size
        elif hasattr(self.llm.model.config, "d_model"):
            self.model_dim = self.llm.model.config.d_model
        else:
            raise ValueError("Could not determine model dimension from LLM config.")

        if probe is None:
            self.probe = Probe(self.model_dim)
        else:
            self.probe = probe

        # BCELoss because self.probe outputs confidence probabilities in [0, 1]
        self.loss_fn = nn.BCELoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generate: bool = True,
    ):
        """
        Run forward pass of LLM to generate response, extract activations, and run Probe to get confidence.
        
        Returns:
            If generate is True: (generated_ids, confidence)
            If generate is False: confidence
        """
        llm_output = self.llm(
            input_ids,
            attention_mask,
            generate=generate,
            return_hidden_states=True,
        )

        if generate:
            generated_ids, hidden_states = llm_output
        else:
            hidden_states = llm_output

        activations = hidden_states[self.layer_idx]
        confidence = self.probe(activations)

        if generate:
            return generated_ids, confidence
        return confidence

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        correct = batch["correct"]

        # Call forward with generate=False to extract confidence directly and avoid repetition
        confidence = self(input_ids, attention_mask, generate=False)
        loss = self.loss_fn(confidence, correct.float())

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        correct = batch["correct"]

        # Call forward with generate=False to extract confidence directly and avoid repetition
        confidence = self(input_ids, attention_mask, generate=False)
        loss = self.loss_fn(confidence, correct.float())

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return {"val_loss": loss, "confidence": confidence, "correct": correct}

    def configure_optimizers(self):
        # Only self.probe parameters are optimized
        return torch.optim.AdamW(self.probe.parameters(), lr=self.lr)
