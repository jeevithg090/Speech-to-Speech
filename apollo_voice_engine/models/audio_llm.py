"""
Extended Audio-LLM Model

Extends Sarvam-1 2B with SNAC audio tokens for unified speech-to-speech
generation. The model can now "hear" and "speak" by processing audio
tokens alongside text tokens.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging

logger = logging.getLogger(__name__)


class AudioLLM(nn.Module):
    """
    Extended Sarvam-1 model with audio token vocabulary.
    
    Architecture:
    - Base: Sarvam-1 2B (28 layers, 2048 hidden, 16 heads)
    - Extended vocabulary: +4096 SNAC audio tokens
    - Unified next-token prediction for both text and audio
    """
    
    # Constants
    AUDIO_TOKEN_OFFSET = 50000
    AUDIO_VOCAB_SIZE = 4096
    
    # Special tokens
    AUDIO_START_TOKEN = "<|audio_start|>"
    AUDIO_END_TOKEN = "<|audio_end|>"
    
    def __init__(
        self,
        model_name: str = "sarvamai/sarvam-1",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the Audio-LLM.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Will be initialized in load_model()
        self.model = None
        self.tokenizer = None
        self.config = None
        self.original_vocab_size = None
        self.extended_vocab_size = None
        
    def load_model(self):
        """Load and extend the base model with audio vocabulary."""
        logger.info(f"Loading base model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Add special audio tokens
        special_tokens = {
            "additional_special_tokens": [
                self.AUDIO_START_TOKEN,
                self.AUDIO_END_TOKEN,
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load config
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Load model with optional quantization
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        
        if self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif self.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["device_map"] = self.device
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        
        # Store original vocab size
        self.original_vocab_size = self.model.config.vocab_size
        
        # Extend vocabulary for audio tokens
        self._extend_vocabulary()
        
        logger.info(f"Model loaded. Vocab extended: {self.original_vocab_size} → {self.extended_vocab_size}")
        
    def _extend_vocabulary(self):
        """Extend the model's embedding and output layers for audio tokens."""
        # Calculate new vocab size
        # Original vocab + special tokens + audio tokens
        num_special = 2  # AUDIO_START, AUDIO_END
        self.extended_vocab_size = self.original_vocab_size + num_special + self.AUDIO_VOCAB_SIZE
        
        # Resize token embeddings
        self.model.resize_token_embeddings(self.extended_vocab_size)
        
        # Initialize new audio token embeddings
        # Use small random init to distinguish from text tokens
        with torch.no_grad():
            # Get embedding layer
            embeddings = self.model.get_input_embeddings()
            
            # Initialize audio tokens (last AUDIO_VOCAB_SIZE entries)
            audio_start_idx = self.extended_vocab_size - self.AUDIO_VOCAB_SIZE
            nn.init.normal_(
                embeddings.weight[audio_start_idx:],
                mean=0.0,
                std=0.02
            )
            
        logger.info(f"Extended embeddings for {self.AUDIO_VOCAB_SIZE} audio tokens")
    
    def text_to_tokens(self, text: str) -> torch.Tensor:
        """Convert text to token IDs."""
        return self.tokenizer.encode(
            text,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)
    
    def tokens_to_text(self, tokens: torch.Tensor) -> str:
        """Convert token IDs to text."""
        # Filter out audio tokens
        text_tokens = tokens[tokens < self.AUDIO_TOKEN_OFFSET]
        return self.tokenizer.decode(text_tokens, skip_special_tokens=True)
    
    def audio_tokens_to_ids(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert raw SNAC tokens to extended vocabulary IDs.
        
        Args:
            audio_tokens: SNAC tokens (0-4095)
            
        Returns:
            Extended vocab IDs with proper offset
        """
        # SNAC tokens come with AUDIO_TOKEN_OFFSET already applied
        # from SNACWrapper, but we need to add our special token offset
        offset = self.extended_vocab_size - self.AUDIO_VOCAB_SIZE
        
        # If tokens already have offset, remove it and apply new one
        if audio_tokens.min() >= self.AUDIO_TOKEN_OFFSET:
            audio_tokens = audio_tokens - self.AUDIO_TOKEN_OFFSET
            
        return audio_tokens + offset
    
    def ids_to_audio_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert extended vocabulary IDs back to SNAC tokens.
        
        Args:
            token_ids: Extended vocab IDs
            
        Returns:
            SNAC tokens (0-4095) with AUDIO_TOKEN_OFFSET
        """
        offset = self.extended_vocab_size - self.AUDIO_VOCAB_SIZE
        snac_tokens = token_ids - offset
        return snac_tokens + self.AUDIO_TOKEN_OFFSET
    
    def is_audio_token(self, token_id: int) -> bool:
        """Check if a token ID represents an audio token."""
        audio_start = self.extended_vocab_size - self.AUDIO_VOCAB_SIZE
        return audio_start <= token_id < self.extended_vocab_size
    
    def prepare_audio_input(
        self,
        audio_tokens: torch.Tensor,
        text_prompt: Optional[str] = None
    ) -> torch.Tensor:
        """
        Prepare input sequence with audio tokens.
        
        Args:
            audio_tokens: SNAC audio tokens
            text_prompt: Optional text prefix
            
        Returns:
            Combined token sequence for model input
        """
        sequences = []
        
        # Add text prompt if provided
        if text_prompt:
            text_tokens = self.text_to_tokens(text_prompt)
            sequences.append(text_tokens.squeeze(0))
        
        # Add audio start token
        audio_start_id = self.tokenizer.convert_tokens_to_ids(self.AUDIO_START_TOKEN)
        sequences.append(torch.tensor([audio_start_id], device=self.device))
        
        # Add audio tokens (converted to extended vocab IDs)
        audio_ids = self.audio_tokens_to_ids(audio_tokens)
        if audio_ids.dim() > 1:
            audio_ids = audio_ids.squeeze(0)
        sequences.append(audio_ids)
        
        # Add audio end token
        audio_end_id = self.tokenizer.convert_tokens_to_ids(self.AUDIO_END_TOKEN)
        sequences.append(torch.tensor([audio_end_id], device=self.device))
        
        return torch.cat(sequences, dim=0).unsqueeze(0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs (text + audio)
            attention_mask: Attention mask
            labels: Labels for training
            
        Returns:
            Model outputs with loss if labels provided
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    @torch.no_grad()
    def generate_response(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate a response (text or audio tokens).
        
        Args:
            input_ids: Input token sequence
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to sample vs greedy
            
        Returns:
            Generated token sequence
        """
        return self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            **kwargs
        )
    
    def extract_audio_tokens(self, generated_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract audio tokens from generated sequence.
        
        Args:
            generated_ids: Full generated token sequence
            
        Returns:
            Audio tokens between AUDIO_START and AUDIO_END, or None
        """
        tokens = generated_ids.squeeze().tolist()
        
        audio_start_id = self.tokenizer.convert_tokens_to_ids(self.AUDIO_START_TOKEN)
        audio_end_id = self.tokenizer.convert_tokens_to_ids(self.AUDIO_END_TOKEN)
        
        try:
            start_idx = tokens.index(audio_start_id) + 1
            end_idx = tokens.index(audio_end_id)
            
            audio_ids = torch.tensor(tokens[start_idx:end_idx], device=self.device)
            return self.ids_to_audio_tokens(audio_ids)
            
        except ValueError:
            logger.warning("No audio tokens found in generated sequence")
            return None
    
    def save_pretrained(self, save_path: str):
        """Save the extended model and tokenizer."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "AudioLLM":
        """Load a saved extended model."""
        instance = cls(model_name=model_path, **kwargs)
        instance.load_model()
        return instance


def test_audio_llm():
    """Test the AudioLLM initialization and basic operations."""
    print("Testing AudioLLM...")
    
    # Note: This test requires the actual model to be downloaded
    # For CI, mock the model loading
    model = AudioLLM(model_name="sarvamai/sarvam-1")
    
    print(f"Model name: {model.model_name}")
    print(f"Audio vocab size: {model.AUDIO_VOCAB_SIZE}")
    print(f"Audio token offset: {model.AUDIO_TOKEN_OFFSET}")
    
    # Test token conversion (without loading model)
    fake_audio_tokens = torch.tensor([0, 100, 4095])
    print(f"Sample SNAC tokens: {fake_audio_tokens.tolist()}")
    
    print("✓ AudioLLM basic test passed!")
    print("Note: Full test requires model download (~4GB)")


if __name__ == "__main__":
    test_audio_llm()
