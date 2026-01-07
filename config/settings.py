"""
Configuration Management for Deep Research Agent
Handles environment variables, API keys, and runtime settings.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import json


class LLMProvider(Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class SearchProvider(Enum):
    """Supported search providers"""
    SERPER = "serper"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SCRAPER = "scraper"


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: LLMProvider = LLMProvider.GEMINI
    model: str = "gemini-2.0-flash"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 8192
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


@dataclass
class SearchConfig:
    """Search engine configuration"""
    providers: List[SearchProvider] = field(default_factory=lambda: [SearchProvider.DUCKDUCKGO])
    serper_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    max_results_per_query: int = 10
    max_concurrent_searches: int = 5
    timeout: int = 30
    enable_academic_search: bool = True
    enable_local_search: bool = True
    
    def __post_init__(self):
        self.serper_api_key = self.serper_api_key or os.getenv("SERPER_API_KEY")
        self.tavily_api_key = self.tavily_api_key or os.getenv("TAVILY_API_KEY")


@dataclass
class ReasoningConfig:
    """Tree-of-Thoughts reasoning configuration"""
    max_depth: int = 5
    branching_factor: int = 3
    pruning_threshold: float = 0.3
    confidence_threshold: float = 0.7
    max_iterations: int = 50
    enable_parallel_exploration: bool = True
    enable_dynamic_depth: bool = True
    backtrack_on_low_confidence: bool = True


@dataclass
class MemoryConfig:
    """Memory system configuration"""
    short_term_capacity: int = 100
    long_term_backend: str = "chromadb"  # chromadb, faiss, or json
    vector_dimensions: int = 768
    similarity_threshold: float = 0.75
    enable_persistence: bool = True
    storage_path: Path = field(default_factory=lambda: Path("./data/memory"))
    failure_memory_ttl_hours: int = 24


@dataclass 
class VerificationConfig:
    """Verification and fact-checking configuration"""
    min_sources_for_verification: int = 2
    hallucination_threshold: float = 0.5
    contradiction_sensitivity: float = 0.7
    enable_self_critique: bool = True
    max_verification_attempts: int = 3


@dataclass
class CostConfig:
    """Cost and rate limiting configuration"""
    max_api_cost_per_task: float = 1.0  # USD
    max_requests_per_minute: int = 60
    max_tokens_per_task: int = 100000
    enable_cost_tracking: bool = True
    warn_at_cost_percent: float = 0.8


@dataclass
class Settings:
    """Main settings container"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    
    # General settings
    debug: bool = False
    log_level: str = "INFO"
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables"""
        settings = cls()
        
        # Override from env
        if os.getenv("DEBUG"):
            settings.debug = os.getenv("DEBUG", "").lower() in ("true", "1", "yes")
        if os.getenv("LOG_LEVEL"):
            settings.log_level = os.getenv("LOG_LEVEL", "INFO")
            
        return settings
    
    @classmethod
    def from_json(cls, path: Path) -> "Settings":
        """Load settings from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: Path) -> None:
        """Save settings to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._to_dict(), f, indent=2)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        def convert(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            return obj
        return convert(self)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def configure(settings: Settings) -> None:
    """Set global settings"""
    global _settings
    _settings = settings
