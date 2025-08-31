"""
LLM Processor v1.0 - AI-Powered Script Analysis Engine
Uses Google Gemini AI for comprehensive transcript analysis including:
- Lead character emotion profiling
- Keyword extraction with percentages
- Overall sentiment classification
- Content genre classification
- Contextual ad placement recommendations
- Script synopsis generation
- Metadata visualization data
"""

import json
import os
import re
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Environment for API key
from dotenv import load_dotenv
load_dotenv()

class ScriptAnalysisResult(BaseModel):
    """Pydantic model for structured LLM output"""
    
    # Script metadata
    script_metadata: Dict[str, Any] = Field(description="Basic script information")
    
    # Lead characters with emotions
    lead_characters: List[Dict[str, Any]] = Field(description="Top 5-10 characters with emotion analysis")
    
    # Keywords and topics
    top_keywords: List[Dict[str, Any]] = Field(description="Most frequent keywords with percentages")
    
    # Overall sentiment
    overall_sentiment: Dict[str, Any] = Field(description="Script-wide sentiment analysis")
    
    # Content classification
    content_classification: Dict[str, Any] = Field(description="Genre and content type classification")
    
    # Ad placement recommendations
    ad_placement_recommendations: Dict[str, Any] = Field(description="Contextual ad placement suggestions")
    
    # Script synopsis
    script_synopsis: str = Field(description="Brief movie/content summary")
    
    # Named entity recognition summary
    named_entity_summary: Dict[str, Any] = Field(description="Entity types and counts for metadata cards")

class LLMProcessor:
    """
    Advanced LLM-powered script analyzer using Google Gemini AI
    Processes complete transcripts for comprehensive metadata extraction
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM processor with Gemini AI"""
        
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or provided as parameter")
        
        # Initialize Gemini LLM with updated model name
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Updated model name
            google_api_key=self.api_key,
            temperature=0.3,  # Low temperature for consistent analysis
            max_tokens=4096
        )
        
        BASE_DIR = Path(__file__).parent  # assumes this script is in project root

        # Setup paths (keep the same folder names)
        self.input_path = Path("data/processed/LLM_jsons")
        self.output_path = Path("data/processed/processed_llm_analyzer_jsons")

        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup output parser
        self.parser = PydanticOutputParser(pydantic_object=ScriptAnalysisResult)
        
        # Create the comprehensive analysis prompt
        self.create_analysis_prompt()
    
    def create_analysis_prompt(self):
        """Create the LangChain prompt template for comprehensive script analysis"""
        
        prompt_template = """
You are an expert AI script analyst specializing in comprehensive media content analysis. Analyze the provided movie/TV script transcript and extract detailed insights.

**ANALYSIS REQUIREMENTS:**

1. **LEAD CHARACTER ANALYSIS** (Top 5-10 characters by importance):
   - Identify main characters based on dialogue frequency and narrative significance
   - For each lead character, determine their DOMINANT emotion throughout the script
   - Classify emotions as: joy, sadness, anger, fear, surprise, trust, neutral
   - Provide confidence scores (0-1) for emotion classifications
   - Include character importance percentage based on mentions/dialogue

2. **KEYWORD & TOPIC EXTRACTION**:
   - Extract top 15-20 most significant keywords/topics from the script
   - Dont include any character names as a keyword
   - Categorize keywords into themes, objects, locations, activities, concepts
   - Calculate occurrence frequency and percentage of total content
   - Focus on themes, objects, locations, activities, and important concepts
   - Exclude common words and focus on meaningful content markers

3. **OVERALL SENTIMENT CLASSIFICATION**:
   - Analyze the entire script's emotional tone
   - Provide percentage breakdown: positive, negative, neutral
   - Give overall classification with confidence score
   - Describe sentiment flow (consistent, improving, declining, mixed)

4. **CONTENT CLASSIFICATION**:
   - Identify primary genres (action, drama, comedy, romance, thriller, horror, etc.)
   - Provide confidence scores for each genre
   - List supporting evidence/keywords for each classification

5. **CONTEXTUAL AD PLACEMENT RECOMMENDATIONS**:
   - Identify 5-7 optimal ad placement opportunities
   - For each placement, specify:
     * Scene context and why it's suitable for ads
     * Recommended ad types based on scene content (sports brands, food, cars, etc.)
     * Timing preference (if timestamps available) or scene description
     * Suitability score (0-1)
   - Focus on natural breaks, scene transitions, low-intensity moments

6. **SCRIPT SYNOPSIS**:
   - Write a concise 2-3 sentence summary of the movie/content
   - Include main plot, key characters, and genre
   - Make it informative for users to understand the content type

7. **NAMED ENTITY RECOGNITION METADATA**:
   - Count and categorize entities: PERSON, LOCATION, ORGANIZATION
   - Provide totals for each entity type for metadata card display
   - Include top 3-5 entities in each category

**OUTPUT FORMAT REQUIREMENTS:**
Return a valid JSON structure matching this exact format:

{{
  "script_metadata": {{
    "total_words": 30000,
    "estimated_duration_minutes": 120,
    "analysis_timestamp": "2025-08-30T14:30:00",
    "processing_model": "gemini-pro"
  }},
  "lead_characters": [
    {{
      "name": "character_name",
      "dominant_emotion": "emotion",
      "emotion_confidence": 0.85,
      "importance_percentage": 25.5,
      "total_mentions": 45,
      "character_description": "brief_role_description"
    }}
  ],
  "top_keywords": [
    {{
      "keyword": "keyword",
      "frequency": 23,
      "percentage": 2.5,
      "category": "theme"
    }}
  ],
  "overall_sentiment": {{
    "classification": "positive",
    "confidence": 0.75,
    "distribution": {{
      "positive": 45,
      "negative": 25,
      "neutral": 30
    }},
    "sentiment_flow": "consistent",
    "key_emotional_moments": ["description1", "description2"]
  }},
  "content_classification": {{
    "primary_genres": [
      {{
        "genre": "Action",
        "confidence": 0.85,
        "supporting_keywords": ["fight", "gun", "chase"]
      }}
    ],
    "content_rating_suggestion": "R",
    "target_audience": "Adults 18-35"
  }},
  "ad_placement_recommendations": {{
    "total_recommended_slots": 5,
    "optimal_placements": [
      {{
        "placement_id": 1,
        "scene_context": "scene_description",
        "timing_description": "when_to_place_ad",
        "suitability_score": 0.85,
        "recommended_ad_types": ["ad_type1", "ad_type2"],
        "reasoning": "why_this_placement_works",
        "timestamp_estimate": "00:15:30"
      }}
    ]
  }},
  "script_synopsis": "Brief 2-3 sentence movie summary here",
  "named_entity_summary": {{
    "person_count": 15,
    "location_count": 8,
    "organization_count": 3,
    "top_entities": {{
      "persons": ["name1", "name2", "name3"],
      "locations": ["location1", "location2", "location3"],
      "organizations": ["org1", "org2", "org3"]
    }}
  }}
}}

**ANALYSIS INSTRUCTIONS:**
- Analyze the COMPLETE script comprehensively
- Focus on CHARACTER EMOTIONS and their evolution through the story
- Identify CONTEXTUAL AD OPPORTUNITIES that match scene content
- Provide ACCURATE PERCENTAGES and CONFIDENCE SCORES
- Ensure JSON format is valid and complete
- Make recommendations PRACTICAL and ACTIONABLE

**SCRIPT TO ANALYZE:**
{transcript_text}

Provide your comprehensive analysis in the specified JSON format:
"""
        
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def load_transcript(self, file_path: str) -> str:
        """Load and extract text from LLM JSON file"""
        try:
            print(f"📂 Loading transcript from: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            print(f"📊 Loaded JSON structure: {type(data)}")
            
            # Extract text from LLM JSON format
            if isinstance(data, list):
                print(f"📋 Processing list with {len(data)} items")
                # If it's a list of chunks
                text_parts = []
                for i, chunk in enumerate(data):
                    if isinstance(chunk, dict):
                        # Look for text in common fields
                        text = (chunk.get('text', '') or 
                               chunk.get('content', '') or 
                               chunk.get('transcript', '') or
                               chunk.get('dialogue', '') or
                               chunk.get('script', ''))
                        if text:
                            text_parts.append(str(text))
                        else:
                            # Try to get any string value from the chunk
                            for key, value in chunk.items():
                                if isinstance(value, str) and len(value) > 20:
                                    text_parts.append(value)
                                    break
                
                combined_text = ' '.join(text_parts)
                print(f"📝 Extracted {len(combined_text)} characters from list format")
                return combined_text
            
            elif isinstance(data, dict):
                print("📋 Processing dictionary format")
                # If it's a single object, try multiple extraction methods
                possible_fields = ['transcript', 'text', 'content', 'script', 'dialogue', 'full_text']
                
                for field in possible_fields:
                    if field in data:
                        content = data[field]
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict):
                                    text = (item.get('text', '') or 
                                           item.get('content', '') or 
                                           item.get('dialogue', ''))
                                    if text:
                                        text_parts.append(str(text))
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            result = ' '.join(text_parts)
                            if result:
                                print(f"📝 Extracted {len(result)} characters from {field} field")
                                return result
                        elif isinstance(content, str):
                            print(f"📝 Extracted {len(content)} characters from {field} field")
                            return content
                
                # If no standard fields found, try to concatenate all string values
                all_text = []
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 50:
                        all_text.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and len(item) > 20:
                                all_text.append(item)
                
                if all_text:
                    result = ' '.join(all_text)
                    print(f"📝 Extracted {len(result)} characters from concatenated fields")
                    return result
            
            # Fallback: convert entire data to string
            fallback_text = str(data)
            print(f"⚠️ Using fallback extraction: {len(fallback_text)} characters")
            return fallback_text
            
        except Exception as e:
            print(f"❌ Error loading transcript from {file_path}: {str(e)}")
            print(f"🔍 Error details: {traceback.format_exc()}")
            return ""
    
    def chunk_text(self, text: str, max_chunk_size: int = 15000) -> List[str]:
        """Split large text into manageable chunks for LLM processing"""
        
        if len(text) <= max_chunk_size:
            return [text]
        
        # Split by paragraphs first, then by sentences if needed
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= max_chunk_size:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph + '\n\n'
                else:
                    # Single paragraph too long, split by sentences
                    sentences = re.split(r'[.!?]+', paragraph)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= max_chunk_size:
                            current_chunk += sentence + '. '
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + '. '
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def analyze_script_with_llm(self, transcript_text: str) -> Dict[str, Any]:
        """Perform comprehensive script analysis using Gemini LLM"""
        
        try:
            print("🤖 Starting LLM analysis with Gemini AI...")
            print(f"📄 Script length: {len(transcript_text)} characters")
            print(f"📝 Word count: {len(transcript_text.split())} words")
            
            # Clean and validate transcript text
            if not transcript_text or len(transcript_text.strip()) < 100:
                raise ValueError("Transcript text is too short or empty")
            
            # For very large scripts, analyze in chunks and then synthesize
            if len(transcript_text) > 20000:
                print("📚 Large script detected, processing in intelligent chunks...")
                chunks = self.chunk_text(transcript_text, max_chunk_size=15000)
                print(f"🔢 Created {len(chunks)} chunks for analysis")
                
                # Analyze first chunk in detail, then summarize remaining chunks
                print("🔍 Analyzing primary chunk...")
                main_analysis = self._analyze_chunk(chunks[0], is_primary=True)
                
                # Quick analysis of remaining chunks for additional context
                additional_insights = []
                for i, chunk in enumerate(chunks[1:3], 1):  # Process up to 3 chunks total
                    print(f"🔍 Processing chunk {i+1}/{min(len(chunks), 3)}...")
                    chunk_insight = self._analyze_chunk(chunk, is_primary=False)
                    if chunk_insight:
                        additional_insights.append(chunk_insight)
                
                # Synthesize final results
                print("🔧 Synthesizing comprehensive analysis...")
                final_result = self._synthesize_analysis(main_analysis, additional_insights, transcript_text)
                
            else:
                # Single analysis for smaller scripts
                print("📖 Processing complete script in single analysis...")
                final_result = self._analyze_chunk(transcript_text, is_primary=True)
            
            # Validate result structure
            if not final_result or not isinstance(final_result, dict):
                raise ValueError("Invalid analysis result structure")
            
            # Add processing metadata
            if "script_metadata" not in final_result:
                final_result["script_metadata"] = {}
            
            final_result["script_metadata"].update({
                "total_words": len(transcript_text.split()),
                "total_characters": len(transcript_text),
                "analysis_timestamp": datetime.now().isoformat(),
                "processing_model": "gemini-1.5-flash",
                "status": "success"
            })
            
            print("✅ LLM analysis completed successfully!")
            return final_result
            
        except Exception as e:
            print(f"❌ Error in LLM analysis: {str(e)}")
            print(f"🔍 Error details: {traceback.format_exc()}")
            return self._create_error_response(str(e))
    
    def _analyze_chunk(self, text: str, is_primary: bool = True) -> Dict[str, Any]:
        """Analyze a single chunk of text with appropriate detail level"""
        
        try:
            print(f"🧠 Sending {len(text)} characters to Gemini AI...")
            
            # Create the prompt with the text
            formatted_prompt = self.prompt.format(transcript_text=text)
            
            # Get LLM response
            print("⏳ Waiting for Gemini AI response...")
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            
            # Extract JSON from response
            response_text = response.content
            print(f"📥 Received response: {len(response_text)} characters")
            
            # Debug: Print first 500 chars of response
            print(f"🔍 Response preview: {response_text[:500]}...")
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                print("✅ JSON structure found in response")
                try:
                    result = json.loads(json_str)
                    print("✅ JSON parsed successfully")
                    return result
                except json.JSONDecodeError as je:
                    print(f"❌ JSON decode error: {str(je)}")
                    # Try to clean and parse again
                    cleaned_json = self._clean_json_response(json_str)
                    print("🔧 Attempting to clean and re-parse JSON...")
                    return json.loads(cleaned_json)
            else:
                print("❌ No JSON structure found in response")
                # If no JSON found, create structured response from text
                return self._parse_text_response(response_text)
                
        except Exception as e:
            print(f"❌ Error analyzing chunk: {str(e)}")
            print(f"🔍 Full error: {traceback.format_exc()}")
            return self._create_error_response(str(e))
    
    def _clean_json_response(self, json_str: str) -> str:
        """Clean and fix common JSON formatting issues from LLM responses"""
        
        # Remove markdown code blocks
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*$', '', json_str)
        
        # Fix common JSON issues
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
        
        # Fix unescaped quotes in strings
        json_str = re.sub(r'(?<!\\)"(?![,\]:}])', '\\"', json_str)
        
        return json_str
    
    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM text response if JSON extraction fails"""
        
        return {
            "script_metadata": {
                "total_words": len(response_text.split()),
                "estimated_duration_minutes": 90,
                "analysis_timestamp": datetime.now().isoformat(),
                "processing_model": "gemini-pro"
            },
            "lead_characters": [
                {
                    "name": "Character Analysis Failed",
                    "dominant_emotion": "neutral",
                    "emotion_confidence": 0.5,
                    "importance_percentage": 0,
                    "total_mentions": 0,
                    "character_description": "Unable to parse character data"
                }
            ],
            "top_keywords": [
                {
                    "keyword": "analysis_error",
                    "frequency": 1,
                    "percentage": 0.1,
                    "category": "error"
                }
            ],
            "overall_sentiment": {
                "classification": "neutral",
                "confidence": 0.5,
                "distribution": {"positive": 33, "negative": 33, "neutral": 34},
                "sentiment_flow": "unknown",
                "key_emotional_moments": ["Analysis parsing failed"]
            },
            "content_classification": {
                "primary_genres": [{"genre": "Unknown", "confidence": 0.5, "supporting_keywords": []}],
                "content_rating_suggestion": "Not Rated",
                "target_audience": "Unknown"
            },
            "ad_placement_recommendations": {
                "total_recommended_slots": 0,
                "optimal_placements": []
            },
            "script_synopsis": "Script analysis failed - unable to generate synopsis",
            "named_entity_summary": {
                "person_count": 0,
                "location_count": 0,
                "organization_count": 0,
                "top_entities": {"persons": [], "locations": [], "organizations": []}
            }
        }
    
    def _synthesize_analysis(self, main_analysis: Dict[str, Any], additional_insights: List[Dict[str, Any]], full_text: str) -> Dict[str, Any]:
        """Synthesize multiple chunk analyses into final comprehensive result"""
        
        try:
            # Use main analysis as base
            final_result = main_analysis.copy()
            
            # Enhance with additional insights
            all_keywords = final_result.get("top_keywords", [])
            all_characters = final_result.get("lead_characters", [])
            
            # Merge keywords from additional chunks
            for insight in additional_insights:
                for keyword in insight.get("top_keywords", []):
                    # Check if keyword already exists
                    existing = next((k for k in all_keywords if k["keyword"] == keyword["keyword"]), None)
                    if existing:
                        existing["frequency"] += keyword["frequency"]
                    else:
                        all_keywords.append(keyword)
            
            # Sort and limit keywords
            all_keywords.sort(key=lambda x: x["frequency"], reverse=True)
            final_result["top_keywords"] = all_keywords[:20]
            
            # Recalculate percentages based on full text
            total_words = len(full_text.split())
            for keyword in final_result["top_keywords"]:
                keyword["percentage"] = round((keyword["frequency"] / total_words) * 100, 2)
            
            # Update metadata
            final_result["script_metadata"]["total_words"] = total_words
            final_result["script_metadata"]["estimated_duration_minutes"] = max(60, total_words // 180)  # ~180 words per minute
            final_result["script_metadata"]["analysis_timestamp"] = datetime.now().isoformat()
            final_result["script_metadata"]["status"] = "synthesized"
            
            return final_result
            
        except Exception as e:
            print(f"Error synthesizing analysis: {str(e)}")
            return main_analysis
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create a default response structure when analysis fails"""
        
        current_time = datetime.now().isoformat()
        
        return {
            "script_metadata": {
                "total_words": 0,
                "estimated_duration_minutes": 0,
                "analysis_timestamp": current_time,
                "processing_model": "gemini-1.5-flash",
                "error": error_message,
                "status": "failed"
            },
            "lead_characters": [],
            "top_keywords": [],
            "overall_sentiment": {
                "classification": "unknown",
                "confidence": 0.0,
                "distribution": {"positive": 0, "negative": 0, "neutral": 100},
                "sentiment_flow": "unknown",
                "key_emotional_moments": []
            },
            "content_classification": {
                "primary_genres": [],
                "content_rating_suggestion": "Not Rated",
                "target_audience": "Unknown"
            },
            "ad_placement_recommendations": {
                "total_recommended_slots": 0,
                "optimal_placements": []
            },
            "script_synopsis": f"Analysis failed: {error_message}",
            "named_entity_summary": {
                "person_count": 0,
                "location_count": 0,
                "organization_count": 0,
                "top_entities": {"persons": [], "locations": [], "organizations": []}
            }
        }
    
    def process_file(self, filename: str) -> bool:
        """Process a single transcript file and save analysis results"""
        
        input_file = self.input_path / filename
        output_file = self.output_path / f"analyzed_{filename}"
        
        if not input_file.exists():
            print(f"❌ File not found: {input_file}")
            return False
        
        if output_file.exists():
            print(f"⚠️  Output file already exists: {output_file}")
            response = input("Overwrite? (y/n): ").lower()
            if response != 'y':
                return False
        
        try:
            print(f"🔄 Processing: {filename}")
            
            # Load transcript
            transcript_text = self.load_transcript(str(input_file))
            if not transcript_text:
                print(f"❌ Could not extract text from {filename}")
                return False
            
            if len(transcript_text.strip()) < 100:
                print(f"❌ Transcript too short: {len(transcript_text)} characters")
                return False
            
            print(f"📊 Loaded {len(transcript_text)} characters, {len(transcript_text.split())} words")
            
            # Show sample of loaded content
            sample = transcript_text[:200] + "..." if len(transcript_text) > 200 else transcript_text
            print(f"📝 Content sample: {sample}")
            
            # Perform LLM analysis
            analysis_result = self.analyze_script_with_llm(transcript_text)
            
            # Validate analysis result
            if not analysis_result or "error" in analysis_result.get("script_metadata", {}):
                print(f"❌ Analysis failed for {filename}")
                return False
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(analysis_result, file, indent=2, ensure_ascii=False)
            
            print(f"✅ Analysis saved to: {output_file}")
            
            # Show analysis summary
            self._print_analysis_summary(analysis_result)
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            print(f"🔍 Error details: {traceback.format_exc()}")
            return False
    
    def process_all_files(self) -> Dict[str, bool]:
        """Process all JSON files in the input directory"""
        
        if not self.input_path.exists():
            print(f"❌ Input directory not found: {self.input_path}")
            return {}
        
        # Find all JSON files
        json_files = list(self.input_path.glob("*.json"))
        
        if not json_files:
            print(f"❌ No JSON files found in: {self.input_path}")
            return {}
        
        print(f"🎬 Found {len(json_files)} transcript files to process")
        
        results = {}
        for json_file in json_files:
            filename = json_file.name
            print(f"\n{'='*50}")
            print(f"🎯 Processing: {filename}")
            print(f"{'='*50}")
            
            success = self.process_file(filename)
            results[filename] = success
            
            if success:
                print(f"✅ {filename} - COMPLETED")
            else:
                print(f"❌ {filename} - FAILED")
        
        return results
    
    def get_analysis_summary(self, results: Dict[str, bool]) -> None:
        """Print a summary of processing results"""
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"\n{'='*60}")
        print(f"🎊 LLM ANALYZER PROCESSING SUMMARY")
        print(f"{'='*60}")
    
    def _print_analysis_summary(self, analysis: Dict[str, Any],successful: int) -> None:
        """Print a quick summary of the analysis results"""
        
        try:
            print(f"\n🎯 ANALYSIS RESULTS:")
            
            # Characters
            characters = analysis.get("lead_characters", [])
            if characters:
                print(f"👥 Lead Characters Found: {len(characters)}")
                for char in characters[:3]:  # Show top 3
                    emotion = char.get("dominant_emotion", "unknown")
                    confidence = char.get("emotion_confidence", 0)
                    print(f"   • {char.get('name', 'Unknown')}: {emotion} ({confidence:.2f} confidence)")
            
            # Keywords
            keywords = analysis.get("top_keywords", [])
            if keywords:
                print(f"🔤 Keywords Extracted: {len(keywords)}")
                for kw in keywords[:5]:  # Show top 5
                    print(f"   • {kw.get('keyword', 'unknown')}: {kw.get('percentage', 0)}%")
            
            # Sentiment
            sentiment = analysis.get("overall_sentiment", {})
            classification = sentiment.get("classification", "unknown")
            confidence = sentiment.get("confidence", 0)
            print(f"😊 Overall Sentiment: {classification} ({confidence:.2f} confidence)")
            
            # Genres
            genres = analysis.get("content_classification", {}).get("primary_genres", [])
            if genres:
                print(f"🎭 Primary Genre: {genres[0].get('genre', 'Unknown')}")
            
            # Ad placements
            ads = analysis.get("ad_placement_recommendations", {})
            slots = ads.get("total_recommended_slots", 0)
            print(f"📺 Ad Placement Opportunities: {slots}")
            
            print(f"✅ Analysis extraction completed successfully!")
            
        except Exception as e:
            print(f"❌ Error printing analysis summary: {str(e)}")
            print(f"🔍 Error details: {traceback.format_exc()}")
        if successful > 0:
            print(f"\n🎯 Analyzed files saved to:")
            print(f"   {self.output_path}")
            print(f"\n🚀 Ready for dashboard visualization!")
        
        print(f"{'='*60}")

def main():
    """Main execution function - updated for integration"""
    
    print("🎬 AI-Powered LLM Processor v1.0")
    print("🤖 Powered by Google Gemini AI")
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found!")
        raise ValueError("GEMINI_API_KEY required for LLM processing")
    
    try:
        # Initialize processor with updated paths
        processor = LLMProcessor()
        
        # Process all files
        results = processor.process_all_files()
        
        # Count successful processes
        successful = sum(1 for success in results.values() if success)
        print(f"✅ LLM Processing completed: {successful}/{len(results)} files processed")
        
        return successful
        
    except Exception as e:
        print(f"❌ LLM processing failed: {str(e)}")
        raise e