"""
LLM Configuration Management
Provides dynamic model configuration without server restart.
"""
import logging
from dataclasses import dataclass
from typing import Optional
from threading import Lock

logger = logging.getLogger(__name__)

# Allowed models for validation
ALLOWED_MODELS = {
    "google/medgemma-4b-it",
    "microsoft/MediPhi",
    "Intelligent-Internet/II-Medical-8B"
}

@dataclass
class LLMConfig:
    """Configuration for LLM initialization."""
    model_name: str = "microsoft/MediPhi"
    temperature: float = 0.05
    max_new_tokens: int = 512  # Increased from 256 to prevent truncation

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate configuration parameters."""
        if self.model_name not in ALLOWED_MODELS:
            return False, f"Model must be one of: {', '.join(ALLOWED_MODELS)}"
        if not (0.0 <= self.temperature <= 1.0):
            return False, "Temperature must be between 0.0 and 1.0"
        if not (64 <= self.max_new_tokens <= 2048):
            return False, "Max tokens must be between 64 and 2048"
        return True, None


class LLMConfigManager:
    """Thread-safe manager for LLM configuration and agent instances."""

    def __init__(self):
        self._config = LLMConfig()
        self._lock = Lock()
        self._cached_agent = None
        self._cached_llm = None
        self._cached_qa_chain = None
        self._cached_agent_executor = None

    def get_config(self) -> LLMConfig:
        """Get current configuration."""
        with self._lock:
            return LLMConfig(
                model_name=self._config.model_name,
                temperature=self._config.temperature,
                max_new_tokens=self._config.max_new_tokens
            )

    def update_config(self, model_name: Optional[str] = None,
                     temperature: Optional[float] = None,
                     max_new_tokens: Optional[int] = None) -> tuple[bool, Optional[str]]:
        """
        Update configuration and invalidate cache.
        Returns (success, error_message).
        """
        with self._lock:
            # Create new config with updates
            new_config = LLMConfig(
                model_name=model_name if model_name is not None else self._config.model_name,
                temperature=temperature if temperature is not None else self._config.temperature,
                max_new_tokens=max_new_tokens if max_new_tokens is not None else self._config.max_new_tokens
            )

            # Validate
            valid, error = new_config.validate()
            if not valid:
                return False, error

            # Check if model is actually changing
            model_changed = new_config.model_name != self._config.model_name
            old_model = self._config.model_name

            # If model is changing, EXPLICITLY DELETE old model to free memory
            if model_changed and self._cached_agent is not None:
                logger.info(f"Model changing: {old_model} → {new_config.model_name}")
                logger.info("Unloading old model to free memory...")

                # Log memory before unload (optional, for debugging)
                try:
                    import psutil
                    import os
                    process = psutil.Process(os.getpid())
                    mem_before = process.memory_info().rss / 1024 / 1024  # MB
                    logger.info(f"Memory before unload: {mem_before:.0f} MB")
                except:
                    pass

                # Explicitly delete cached objects to release memory immediately
                try:
                    # Delete in reverse order of creation
                    if self._cached_agent_executor is not None:
                        del self._cached_agent_executor
                    if self._cached_qa_chain is not None:
                        del self._cached_qa_chain
                    if self._cached_llm is not None:
                        del self._cached_llm
                    if self._cached_agent is not None:
                        del self._cached_agent
                except:
                    pass

                # Clear references
                self._cached_agent_executor = None
                self._cached_qa_chain = None
                self._cached_llm = None
                self._cached_agent = None

                # Force aggressive garbage collection
                import gc
                gc.collect()
                gc.collect()  # Run twice to catch cyclical references
                gc.collect()  # Third time to be absolutely sure

                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Wait for GPU operations to complete
                        logger.info("Cleared CUDA cache")
                except:
                    pass

                # Log memory after unload (optional, for debugging)
                try:
                    process = psutil.Process(os.getpid())
                    mem_after = process.memory_info().rss / 1024 / 1024  # MB
                    mem_freed = mem_before - mem_after
                    logger.info(f"Memory after unload: {mem_after:.0f} MB (freed: {mem_freed:.0f} MB)")
                except:
                    pass

                logger.info(f"Old model unloaded. Memory freed. New model will load on next request.")
            else:
                # Just parameter change, clear cache
                self._cached_agent = None
                self._cached_llm = None
                self._cached_qa_chain = None
                self._cached_agent_executor = None
                logger.info(f"LLM parameters updated: {self._config}")

            # Apply new configuration
            self._config = new_config

            return True, None

    def get_or_create_agent(self, search_tool):
        """
        Get cached LLMAgent or create a new one based on current config.
        Returns (llm_agent, llm, qa_chain, agent_executor).
        """
        with self._lock:
            if self._cached_agent is None:
                from part_DEF_agent_llm_capabilities import LLMAgent

                logger.info(f"Creating new LLMAgent with config: {self._config}")

                # Create agent with current config
                self._cached_agent = LLMAgent(
                    model_name=self._config.model_name,
                    max_new_tokens=self._config.max_new_tokens,
                    temperature=self._config.temperature
                )

                # Build LangChain components
                self._cached_llm = self._cached_agent.load_llm_lc()
                self._cached_qa_chain = self._cached_agent.build_lc_qa_chain(
                    self._cached_llm, search_tool
                ) if search_tool else None
                self._cached_agent_executor = self._cached_agent.build_agent_executor(
                    self._cached_llm, search_tool, max_iterations=5
                )

                logger.info("LLMAgent and chains created successfully")

            return (
                self._cached_agent,
                self._cached_llm,
                self._cached_qa_chain,
                self._cached_agent_executor
            )


# Global singleton instance
_config_manager = LLMConfigManager()

def get_config_manager() -> LLMConfigManager:
    """Get the global configuration manager instance."""
    return _config_manager
