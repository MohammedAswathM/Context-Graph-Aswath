from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Neo4jSettings(BaseSettings):
    """Neo4j database configuration"""
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    username: str = Field(default="neo4j", description="Neo4j username")
    password: str = Field(default="password", description="Neo4j password")
    database: str = Field(default="contextgraph", description="Neo4j database name")
    max_connection_pool_size: int = Field(default=50, description="Max connections")
    connection_timeout: int = Field(default=30, description="Connection timeout (seconds)")
    
    model_config = SettingsConfigDict(
        env_prefix='NEO4J_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )


class LLMSettings(BaseSettings):
    """LLM configuration - Supports OpenAI and Google Gemini"""
    provider: Literal["openai", "gemini"] = Field(
        default="openai",
        description="LLM provider - 'openai' or 'gemini'"
    )
    model: str = Field(
        default="gpt-3.5-turbo",
        description="Model name (gpt-3.5-turbo for OpenAI, gemini-pro for Gemini)"
    )
    # API Keys - only one is required based on provider
    openai_api_key: str = Field(default="", description="OpenAI API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    
    # Model parameters
    temperature: float = Field(default=0.0, description="LLM temperature")
    max_tokens: int = Field(default=8000, description="Max response tokens")
    timeout: int = Field(default=60, description="Request timeout (seconds)")
    
    model_config = SettingsConfigDict(
        env_prefix='DEFAULT_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )


class ProcessingSettings(BaseSettings):
    """Text processing configuration"""
    chunk_size: int = Field(default=1000, description="Characters per chunk")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    max_concurrent_extractions: int = Field(
        default=5,
        description="Max parallel extractions"
    )
    
    model_config = SettingsConfigDict(
        env_prefix='',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )


class AppSettings(BaseSettings):
    """Application settings"""
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Component settings
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )


# Singleton instance
_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings