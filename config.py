"""
Configuration settings for AI-Powered Metadata Tagging System
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from typing import Dict


# Load environment variables
load_dotenv()

class Settings:
    """Application settings and configuration"""
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    
    # File paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    
    # Processing directories
    LLM_JSONS_DIR: Path = PROCESSED_DIR / "LLM_jsons"
    NLP_JSONS_DIR: Path = PROCESSED_DIR / "NLP_jsons"
    LLM_RESULTS_DIR: Path = PROCESSED_DIR / "processed_llm_analyzer_jsons"
    NLP_RESULTS_DIR: Path = PROCESSED_DIR / "processed_nlp_validator_jsons"
    CROSS_VALIDATOR_DIR: Path = PROCESSED_DIR / "cross_validator"
    VISUALIZATION_DIR: Path = PROCESSED_DIR / "visualization_engine"
    
    # File upload settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: set = {'.txt'}
    
    # Processing settings
    MAX_CONCURRENT_PROCESSES: int = 1
    PROCESSING_TIMEOUT: int = 600  # 10 minutes
    
    # LLM settings
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 4096
    LLM_MODEL: str = "gemini-1.5-flash"
    
    # NLP settings
    SPACY_MODEL: str = "en_core_web_sm"
    USE_VADER_SENTIMENT: bool = True
    TOP_KEYWORDS_COUNT: int = 15
    TOP_CHARACTERS_COUNT: int = 10
    
    # Visualization settings
    MAX_CHART_ITEMS: int = 10
    DEFAULT_CHART_COLORS: List[str] = [
        "#3498db", "#e74c3c", "#2ecc71", "#f39c12", 
        "#9b59b6", "#1abc9c", "#34495e", "#e67e22"
    ]
    
    def __post_init__(self):
        """Create directories on initialization"""
        directories = [
            self.UPLOADS_DIR,
            self.LLM_JSONS_DIR,
            self.NLP_JSONS_DIR,
            self.LLM_RESULTS_DIR,
            self.NLP_RESULTS_DIR,
            self.CROSS_VALIDATOR_DIR,
            self.VISUALIZATION_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are present"""
        return {
            "gemini_key_present": bool(self.GEMINI_API_KEY),
            "huggingface_key_present": bool(self.HUGGINGFACE_API_KEY),
            "ready_for_processing": bool(self.GEMINI_API_KEY)  # Gemini is required
        }

# Create settings instance
settings = Settings()

# Validate configuration on import
api_key_status = settings.validate_api_keys()
if not api_key_status["ready_for_processing"]:
    print("⚠️  WARNING: GEMINI_API_KEY not found in environment variables!")
    print("💡 Please add your API key to .env file:")
    print("   GEMINI_API_KEY=your_gemini_api_key_here")
    print("🔗 Get your API key from: https://makersuite.google.com/app/apikey")

# Create directories
settings.__post_init__()