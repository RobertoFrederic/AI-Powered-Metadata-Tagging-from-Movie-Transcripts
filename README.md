# AI-Powered Metadata Tagging System

## Project Overview

An intelligent system that automatically extracts rich metadata from movie/TV transcripts using dual AI validation (Google Gemini AI + Local NLP). The system provides comprehensive script analysis including character emotions, keyword extraction, sentiment analysis, genre classification, and smart advertisement placement recommendations.

## Key Features

- **Dual Validation Architecture**: Combines Gemini AI analysis with local NLP processing for maximum accuracy
- **Character Emotion Profiling**: Identifies lead characters and their dominant emotions throughout the script
- **Smart Keyword Extraction**: Extracts meaningful keywords with frequency percentages
- **Sentiment Analysis**: Script-wide emotional tone classification with confidence scores
- **Content Classification**: Automatic genre identification (action, drama, comedy, etc.)
- **Ad Placement Intelligence**: Context-aware advertisement insertion recommendations
- **Comprehensive Entity Recognition**: Persons, locations, organizations with importance ranking

## Project Architecture

```
AI-Metadata-Tagging/
├── data/
│   ├── uploads/                                    # Raw transcript files (.txt)
│   └── processed/
│       ├── LLM_jsons/                             # Preprocessed for Gemini AI
│       ├── NLP_jsons/                             # Preprocessed for local NLP
│       ├── processed_llm_analyzer_jsons/          # Gemini AI analysis results
│       ├── processed_nlp_validator_jsons/         # Local NLP analysis results
│       └── script_analysis_results/               # Additional analysis outputs
├── file_handler.py                                # Transcript preprocessing
├── nlp_validator.py                               # Local NLP analysis engine
├── llm_processor.py                               # Gemini AI analysis engine
├── requirements.txt                               # Python dependencies
├── .env                                          # API keys (create this)
└── README.md                                     # This file
```

## Technology Stack

- **AI Models**: Google Gemini 1.5 Flash, spaCy, VADER Sentiment
- **Backend**: Python 3.8+, LangChain, Pydantic
- **NLP Processing**: spaCy, TextBlob, Transformers
- **Data Processing**: Pandas, NumPy, JSON

## Installation & Setup

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd ai-metadata-tagging

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

### 2. Create Directory Structure

```bash
# Create required directories
mkdir -p data/uploads
mkdir -p data/processed/LLM_jsons
mkdir -p data/processed/NLP_jsons
mkdir -p data/processed/processed_llm_analyzer_jsons
mkdir -p data/processed/processed_nlp_validator_jsons
mkdir -p data/processed/script_analysis_results
```

### 3. Configure API Keys

Create a `.env` file in the root directory:

```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here  # Optional
```

**Get your Gemini API key**: https://makersuite.google.com/app/apikey

## Usage Instructions

### Step 1: Prepare Transcript Files

Place your raw transcript files (.txt format) in the `data/uploads/` directory.

**Supported formats:**
- Plain text transcripts
- Movie scripts
- TV show transcripts
- Any text-based dialogue content

### Step 2: Preprocess Transcripts

Run the file handler to prepare data for both AI and NLP analysis:

```bash
python file_handler.py
```

**Output:**
- `data/processed/LLM_jsons/` - Optimized for Gemini AI analysis
- `data/processed/NLP_jsons/` - Structured for local NLP processing

### Step 3: Run NLP Analysis

Execute local NLP processing for baseline analysis:

```bash
python nlp_validator.py
```

**Output:**
- `data/processed/processed_nlp_validator_jsons/` - Local analysis results
- Keyword extraction, entity recognition, sentiment analysis
- Fast processing using spaCy and VADER

### Step 4: Run AI Analysis

Execute Gemini AI analysis for comprehensive insights:

```bash
python llm_processor.py
```

**Output:**
- `data/processed/processed_llm_analyzer_jsons/` - AI analysis results
- Character emotion profiling, advanced sentiment analysis
- Context-aware ad placement recommendations
- Script synopsis and genre classification

## Analysis Output Structure

### Gemini AI Analysis Results

```json
{
  "script_metadata": {
    "total_words": 30000,
    "estimated_duration_minutes": 120,
    "analysis_timestamp": "2025-08-30T14:30:00",
    "processing_model": "gemini-1.5-flash"
  },
  "lead_characters": [
    {
      "name": "Character Name",
      "dominant_emotion": "joy",
      "emotion_confidence": 0.85,
      "importance_percentage": 25.5,
      "total_mentions": 45
    }
  ],
  "top_keywords": [
    {
      "keyword": "action",
      "frequency": 23,
      "percentage": 2.5,
      "category": "theme"
    }
  ],
  "overall_sentiment": {
    "classification": "positive",
    "confidence": 0.75,
    "distribution": {
      "positive": 45,
      "negative": 25,
      "neutral": 30
    }
  },
  "content_classification": {
    "primary_genres": [
      {
        "genre": "Action",
        "confidence": 0.85,
        "supporting_keywords": ["fight", "gun", "chase"]
      }
    ]
  },
  "ad_placement_recommendations": {
    "total_recommended_slots": 5,
    "optimal_placements": [
      {
        "placement_id": 1,
        "scene_context": "Scene transition",
        "suitability_score": 0.85,
        "recommended_ad_types": ["automotive", "sports"]
      }
    ]
  }
}
```

### Local NLP Analysis Results

```json
{
  "script_metadata": {
    "total_chunks": 1686,
    "total_words": 45230,
    "processing_time_seconds": 3.45
  },
  "top_keywords": [
    {
      "keyword": "action",
      "frequency": 23,
      "relevance": 0.0051
    }
  ],
  "character_analysis": {
    "lead_characters": [
      {
        "name": "Alice",
        "mentions": 18,
        "percentage": 25.3,
        "importance": "protagonist"
      }
    ]
  },
  "sentiment_analysis": {
    "overall_sentiment": "positive",
    "confidence": 0.725,
    "sentiment_flow": "improving"
  }
}
```

## Advanced Features

### Character Emotion Analysis
- Identifies dominant emotions: joy, sadness, anger, fear, surprise, trust
- Provides confidence scores for each emotion classification
- Tracks character importance based on dialogue frequency

### Smart Ad Placement
- Analyzes scene transitions and content flow
- Recommends optimal timing for advertisement insertion
- Suggests relevant ad types based on scene content
- Provides suitability scores for each placement

### Content Classification
- Identifies primary genres with confidence scores
- Supports 14+ content categories (action, romance, comedy, drama, etc.)
- Provides supporting evidence for each classification

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   Error: GEMINI_API_KEY not found
   ```
   **Solution**: Ensure `.env` file exists with valid API key

2. **spaCy Model Missing**
   ```bash
   OSError: [E050] Can't find model 'en_core_web_sm'
   ```
   **Solution**: Run `python -m spacy download en_core_web_sm`

3. **Memory Issues with Large Files**
   - Files over 20MB may cause memory issues
   - Split large transcripts into smaller chunks
   - Increase system RAM or use chunked processing

4. **JSON Parsing Errors**
   - Check file encoding (use UTF-8)
   - Verify transcript file format
   - Remove null characters or special formatting

### Performance Optimization

- **Small files** (< 10K words): ~30 seconds total processing
- **Medium files** (10K-30K words): ~1-2 minutes total processing  
- **Large files** (> 30K words): ~3-5 minutes total processing

## API Limits & Costs

### Gemini API
- Free tier: 15 requests per minute
- Paid tier: Higher rate limits available
- Monitor usage in Google AI Studio

### Recommended Usage
- Process files during off-peak hours
- Batch process multiple files
- Monitor API quota usage

## Project Status

- ✅ **File Handler**: Preprocessing pipeline complete
- ✅ **NLP Validator**: Local analysis engine complete  
- ✅ **LLM Processor**: Gemini AI integration complete
- 🔄 **Dashboard**: Frontend visualization (next phase)
- 🔄 **API Endpoints**: REST API development (next phase)

## Next Steps

1. **Frontend Dashboard Development**
   - Interactive visualizations for analysis results
   - Real-time processing status
   - Export and sharing capabilities

2. **API Development**
   - RESTful endpoints for web integration
   - Real-time processing status
   - Batch processing capabilities

3. **Advanced Features**
   - Multi-language support
   - Custom model training
   - Video file integration

## Support

For technical support or questions:
1. Check the troubleshooting section above
2. Review log files for detailed error messages
3. Ensure all dependencies are properly installed
4. Verify API keys and internet connectivity

