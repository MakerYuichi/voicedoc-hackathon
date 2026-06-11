"""
Application configuration management using Pydantic Settings
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Google Cloud & Gemini
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    google_cloud_project: str = Field(default="", alias="GOOGLE_CLOUD_PROJECT")
    google_application_credentials: str = Field(default="", alias="GOOGLE_APPLICATION_CREDENTIALS")
    gemini_model: str = Field(default="gemini-1.5-flash-8b", alias="GEMINI_MODEL")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")
    max_tokens: int = Field(default=2048, alias="MAX_TOKENS")
    
    # MongoDB Atlas
    mongodb_uri: str = Field(..., alias="MONGODB_URI")
    mongodb_database: str = Field(default="voicedoc_intelligence", alias="MONGODB_DATABASE")
    
    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    
    # Celery
    celery_broker_url: str = Field(default="redis://localhost:6379/0", alias="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", alias="CELERY_RESULT_BACKEND")
    
    # Application
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # Frontend
    frontend_url: str = Field(default="http://localhost:3000", alias="FRONTEND_URL")
    
    # MongoDB MCP Server
    mcp_server_path: str = Field(default="", alias="MCP_SERVER_PATH")
    mcp_server_port: int = Field(default=5000, alias="MCP_SERVER_PORT")
    
    # Security
    secret_key: str = Field(..., alias="SECRET_KEY")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        alias="CORS_ORIGINS",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated or JSON-array CORS_ORIGINS into a list."""
        val = self.cors_origins.strip()
        if val.startswith("["):
            import json
            return json.loads(val)
        return [o.strip() for o in val.split(",") if o.strip()]
    
    # Agent Settings
    max_parallel_workers: int = Field(default=4, alias="MAX_PARALLEL_WORKERS")
    document_chunk_size: int = Field(default=1000, alias="DOCUMENT_CHUNK_SIZE")
    document_chunk_overlap: int = Field(default=200, alias="DOCUMENT_CHUNK_OVERLAP")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL"
    )
    
    # Vector Search
    vector_index_name: str = Field(default="vector_index", alias="VECTOR_INDEX_NAME")
    # 1536 = Google text-embedding-004 output size
    vector_dimensions: int = Field(default=1536, alias="VECTOR_DIMENSIONS")
    similarity_threshold: float = Field(default=0.7, alias="SIMILARITY_THRESHOLD")
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    @property
    def redis_url(self) -> str:
        """Construct Redis URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


# Global settings instance
settings = Settings()
