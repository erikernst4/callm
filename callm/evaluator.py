"""
Correctness evaluator using an LLM to check semantic equivalence.

Based on the paper's approach: using another LLM to determine if the
predicted answer is semantically equivalent to the ground truth.
"""
from typing import List
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from jinja2 import Template
from callm.config import CACHE_PATH
import re


# Semantic equivalence prompt from the paper
SEMANTIC_EQUIVALENCE_PROMPT = Template("""Are the following two answers to my question Q semantically equivalent?

Q: {{ question }}
A1: {{ gold_answer }}
A2: {{ pred_answer }}

Please answer with a single word, either "Yes." or "No.", and explain your reasoning.""")


class CorrectnessEvaluator:
    """
    LLM-based correctness evaluator.
    
    Uses a separate LLM instance to check if predicted answers are
    semantically equivalent to ground truth answers.
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base", device: str = None):
        """
        Initialize the evaluator.
        
        Args:
            model_name: HuggingFace model name for the evaluator
            device: Device to run on (cuda/cpu). If None, auto-detect.
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Detect model type and load appropriately
        if model_name.startswith('google/flan-t5') or model_name.startswith('t5-'):
            # Seq2Seq model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, 
                cache_dir=CACHE_PATH
            )
            self.is_seq2seq = True
        elif model_name.startswith('Qwen/') or model_name.startswith('meta-llama/'):
            # Causal model  
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=CACHE_PATH,
                trust_remote_code=True if model_name.startswith('Qwen/') else False
            )
            self.is_seq2seq = False
        else:
            # Try seq2seq first, fall back to causal
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=CACHE_PATH
                )
                self.is_seq2seq = True
            except ValueError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=CACHE_PATH
                )
                self.is_seq2seq = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True if model_name.startswith('Qwen/') else False
        )
        
        # Set padding token for causal models
        if not self.is_seq2seq and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        
    def evaluate(
        self, 
        question: str, 
        pred_answer: str, 
        gold_answers: List[str]
    ) -> bool:
        """
        Evaluate if predicted answer is semantically equivalent to any gold answer.
        
        Args:
            question: The original question
            pred_answer: The predicted answer
            gold_answers: List of acceptable ground truth answers
            
        Returns:
            True if the predicted answer is semantically equivalent to any gold answer
        """
        # Try each gold answer
        for gold_answer in gold_answers:
            if self._check_equivalence(question, pred_answer, gold_answer):
                return True
        return False
    
    def _check_equivalence(
        self, 
        question: str, 
        pred_answer: str, 
        gold_answer: str
    ) -> bool:
        """
        Check if two answers are semantically equivalent.
        
        Args:
            question: The original question
            pred_answer: The predicted answer
            gold_answer: One ground truth answer
            
        Returns:
            True if semantically equivalent
        """
        # Generate prompt
        prompt = SEMANTIC_EQUIVALENCE_PROMPT.render(
            question=question,
            gold_answer=gold_answer,
            pred_answer=pred_answer
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,  # Low temperature for more deterministic output
                do_sample=False
            )
        
        # Decode response
        if self.is_seq2seq:
            # Seq2seq: decode the full output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Causal: skip the prompt tokens
            input_length = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Parse response for Yes/No
        # Look for "Yes" at the beginning (case-insensitive)
        response_lower = response.lower().strip()
        if response_lower.startswith("yes"):
            return True
        else:
            return False
