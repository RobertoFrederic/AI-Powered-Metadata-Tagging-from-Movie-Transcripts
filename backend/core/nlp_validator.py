import json
import os
import time
from pathlib import Path
from collections import defaultdict, Counter
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple
import requests
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Configuration
try:
    load_dotenv()
    HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
except:
    HF_API_KEY = None

USE_API = False

print("Starting Script-Level NLP Analysis")
print("Mode: LOCAL-ONLY hybrid approach")
print("Output: Overall script analysis with lead characters, keywords, and recommendations")

# Load models
try:
    nlp = spacy.load("en_core_web_sm")
    print("SpaCy model loaded successfully")
except OSError:
    print("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

try:
    vader_analyzer = SentimentIntensityAnalyzer()
    print("VADER sentiment analyzer loaded")
except Exception as e:
    print(f"VADER not available: {e}")
    vader_analyzer = None

# Keyword and pattern definitions
CONTENT_CATEGORIES = {
    "action": ["action", "fight", "battle", "chase", "explosion", "combat", "war", "attack", "shooting", "punch", "gun", "weapon"],
    "romance": ["love", "kiss", "romantic", "relationship", "date", "heart", "passion", "wedding", "marry", "couple", "romance"],
    "comedy": ["funny", "laugh", "joke", "comedy", "humor", "hilarious", "amusing", "ridiculous", "silly", "comic"],
    "drama": ["drama", "emotional", "tears", "tragic", "serious", "conflict", "tension", "struggle", "pain", "dramatic"],
    "thriller": ["suspense", "mystery", "thriller", "danger", "tension", "twist", "reveal", "secret", "conspiracy"],
    "horror": ["scary", "horror", "fear", "nightmare", "terrifying", "ghost", "monster", "haunted", "evil"],
    "crime": ["crime", "police", "arrest", "criminal", "theft", "robbery", "murder", "investigation", "detective"],
    "family": ["family", "children", "kids", "parents", "home", "mother", "father", "son", "daughter", "brother"],
    "adventure": ["adventure", "journey", "travel", "explore", "quest", "discovery", "expedition", "voyage"],
    "fantasy": ["magic", "wizard", "dragon", "spell", "enchanted", "mythical", "supernatural", "kingdom"],
    "sci-fi": ["space", "alien", "robot", "future", "technology", "spaceship", "planet", "laser", "android"],
    "western": ["cowboy", "sheriff", "saloon", "ranch", "horse", "gunfight", "frontier", "outlaw"],
    "sports": ["game", "team", "player", "coach", "championship", "victory", "defeat", "competition"],
    "business": ["company", "meeting", "office", "boss", "employee", "deal", "contract", "profit", "business"]
}

SENSITIVE_PATTERNS = {
    "violence": ["kill", "murder", "shoot", "stab", "beat", "assault", "violence", "blood", "gore", "torture", "death", "wound"],
    "sexual_content": ["sex", "sexual", "nude", "naked", "intimate", "erotic", "porn", "explicit", "bedroom"],
    "drugs": ["drugs", "cocaine", "heroin", "marijuana", "addiction", "overdose", "dealer", "high", "smoking"],
    "profanity": ["damn", "hell", "shit", "fuck", "bitch", "asshole", "bastard", "crap"],
    "alcohol": ["drunk", "alcohol", "beer", "wine", "vodka", "whiskey", "hangover", "drinking", "bar"]
}

def extract_all_text(transcript: List[Dict]) -> str:
    """Extract all text from transcript chunks"""
    return " ".join([chunk.get('text', '') for chunk in transcript if chunk.get('text')])

def clean_text(text: str) -> str:
    """Clean text for keyword extraction"""
    # Remove stage directions and special formatting
    cleaned = re.sub(r'\[.*?\]', '', text)  # Remove [STAGE DIRECTIONS]
    cleaned = re.sub(r'\(.*?\)', '', cleaned)  # Remove (parenthetical)
    cleaned = re.sub(r'[^a-zA-Z\s]', '', cleaned)  # Remove special characters
    return ' '.join(cleaned.lower().split())

def extract_top_keywords(text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Extract top keywords with frequency"""
    clean = clean_text(text)
    words = clean.split()
    
    # Extended stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'this', 'that',
        'these', 'those', 'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me', 'him', 'her', 'them',
        'us', 'my', 'your', 'his', 'their', 'our', 'up', 'down', 'out', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'back', 'look', 'looks',
        'get', 'gets', 'got', 'go', 'goes', 'went', 'come', 'comes', 'came', 'see', 'sees', 'saw'
    }
    
    word_count = Counter()
    for word in words:
        if len(word) > 2 and word not in stop_words:
            word_count[word] += 1
    
    return [
        {"keyword": word, "frequency": count, "relevance": round(count/len(words), 4)}
        for word, count in word_count.most_common(top_k)
    ]

def extract_all_entities(transcript: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Extract all named entities from entire transcript"""
    if not nlp:
        return {"PER": {}, "LOC": {}, "ORG": {}}
    
    entity_counts = {"PER": Counter(), "LOC": Counter(), "ORG": Counter()}
    
    for chunk in transcript:
        text = chunk.get('text', '')
        if not text:
            continue
            
        doc = nlp(text)
        for ent in doc.ents:
            entity_text = ent.text.strip()
            if len(entity_text) < 2:
                continue
                
            if ent.label_ == "PERSON":
                entity_counts["PER"][entity_text] += 1
            elif ent.label_ in ["GPE", "LOC", "FACILITY"]:
                entity_counts["LOC"][entity_text] += 1
            elif ent.label_ == "ORG":
                entity_counts["ORG"][entity_text] += 1
    
    return {
        category: dict(counter.most_common(20))
        for category, counter in entity_counts.items()
    }

def identify_lead_characters(entities: Dict[str, Dict[str, int]], top_k: int = 10) -> List[Dict[str, Any]]:
    """Identify lead characters based on mention frequency"""
    characters = entities.get("PER", {})
    
    if not characters:
        return []
    
    total_mentions = sum(characters.values())
    lead_chars = []
    
    for char, count in list(characters.items())[:top_k]:
        percentage = (count / total_mentions) * 100
        
        # Classify character importance
        if percentage > 15:
            importance = "protagonist"
        elif percentage > 8:
            importance = "main_character"
        elif percentage > 3:
            importance = "supporting_character"
        else:
            importance = "minor_character"
            
        lead_chars.append({
            "name": char,
            "mentions": count,
            "percentage": round(percentage, 2),
            "importance": importance
        })
    
    return lead_chars

def analyze_overall_sentiment(transcript: List[Dict]) -> Dict[str, Any]:
    """Analyze sentiment across entire script"""
    sentiment_scores = {"positive": [], "neutral": [], "negative": []}
    chunk_sentiments = []
    
    for chunk in transcript:
        text = chunk.get('text', '')
        if not text:
            continue
            
        # Use VADER if available
        if vader_analyzer:
            try:
                scores = vader_analyzer.polarity_scores(text)
                compound = scores['compound']
                
                if compound >= 0.05:
                    label = 'positive'
                elif compound <= -0.05:
                    label = 'negative'
                else:
                    label = 'neutral'
                    
                sentiment_scores[label].append(abs(compound))
                chunk_sentiments.append(label)
                continue
            except:
                pass
        
        # Fallback to TextBlob
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                label = 'positive'
                sentiment_scores[label].append(polarity)
            elif polarity < -0.1:
                label = 'negative'
                sentiment_scores[label].append(abs(polarity))
            else:
                label = 'neutral'
                sentiment_scores[label].append(0.5)
                
            chunk_sentiments.append(label)
        except:
            chunk_sentiments.append('neutral')
    
    # Calculate overall sentiment distribution
    sentiment_counts = Counter(chunk_sentiments)
    total_chunks = len(chunk_sentiments)
    
    if total_chunks == 0:
        return {
            "overall_sentiment": "neutral",
            "confidence": 0.5,
            "distribution": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
            "sentiment_flow": "stable"
        }
    
    distribution = {
        sentiment: round(count / total_chunks, 3)
        for sentiment, count in sentiment_counts.items()
    }
    
    # Determine overall sentiment
    max_sentiment = max(distribution.items(), key=lambda x: x[1])
    overall_sentiment = max_sentiment[0]
    confidence = max_sentiment[1]
    
    # Analyze sentiment flow
    if len(set(chunk_sentiments)) == 1:
        flow = "consistent"
    elif chunk_sentiments[:len(chunk_sentiments)//3].count('positive') > chunk_sentiments[-len(chunk_sentiments)//3:].count('positive'):
        flow = "declining"
    elif chunk_sentiments[:len(chunk_sentiments)//3].count('positive') < chunk_sentiments[-len(chunk_sentiments)//3:].count('positive'):
        flow = "improving"
    else:
        flow = "mixed"
    
    return {
        "overall_sentiment": overall_sentiment,
        "confidence": round(confidence, 3),
        "distribution": distribution,
        "sentiment_flow": flow
    }

def classify_content(text: str) -> List[Dict[str, Any]]:
    """Classify content into categories"""
    text_lower = text.lower()
    category_scores = {}
    
    for category, keywords in CONTENT_CATEGORIES.items():
        score = 0
        matched_keywords = []
        
        for keyword in keywords:
            matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            if matches > 0:
                score += matches
                matched_keywords.append(keyword)
        
        if score > 0:
            confidence = min(0.95, score / (len(text.split()) / 100 + 1))
            category_scores[category] = {
                "score": score,
                "confidence": confidence,
                "matched_keywords": matched_keywords[:5]  # Top 5 matched keywords
            }
    
    # Sort by score and return top categories
    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    
    return [
        {
            "category": category,
            "confidence": round(data["confidence"], 3),
            "relevance_score": data["score"],
            "matched_keywords": data["matched_keywords"]
        }
        for category, data in sorted_categories[:5]  # Top 5 categories
    ]

def detect_sensitive_content_overall(text: str) -> Dict[str, Any]:
    """Detect sensitive content across entire script"""
    text_lower = text.lower()
    
    sensitive_flags = {}
    confidence_scores = {}
    matched_terms = {}
    category_details = {}
    
    for category, patterns in SENSITIVE_PATTERNS.items():
        matches = []
        total_matches = 0
        
        for pattern in patterns:
            pattern_matches = len(re.findall(r'\b' + re.escape(pattern) + r'\b', text_lower))
            if pattern_matches > 0:
                matches.append(pattern)
                total_matches += pattern_matches
        
        if matches:
            sensitive_flags[category] = True
            confidence_scores[category] = min(0.95, (total_matches * 0.2) + (len(matches) * 0.1))
            matched_terms[category] = matches[:5]  # Top 5 matched terms
            category_details[category] = {
                "total_occurrences": total_matches,
                "unique_terms": len(matches),
                "severity": "high" if confidence_scores[category] > 0.7 else "medium" if confidence_scores[category] > 0.4 else "low"
            }
        else:
            sensitive_flags[category] = False
            confidence_scores[category] = 0.0
            category_details[category] = {"total_occurrences": 0, "unique_terms": 0, "severity": "none"}
    
    overall_sensitivity = sum(confidence_scores.values()) / len(confidence_scores)
    
    return {
        "is_sensitive": any(sensitive_flags.values()),
        "overall_sensitivity_score": round(overall_sensitivity, 3),
        "content_warnings": [cat for cat, flag in sensitive_flags.items() if flag],
        "category_analysis": category_details,
        "detailed_flags": sensitive_flags,
        "confidence_scores": {k: round(v, 3) for k, v in confidence_scores.items()}
    }

def extract_timestamps(transcript: List[Dict]) -> List[Dict[str, Any]]:
    """Extract and format all timestamps from transcript"""
    timestamps = []
    
    for i, chunk in enumerate(transcript):
        timestamp_data = chunk.get('timestamp', {})
        
        if isinstance(timestamp_data, dict) and ('start' in timestamp_data or 'end' in timestamp_data):
            start = timestamp_data.get('start', '00:00:00')
            end = timestamp_data.get('end', '00:00:00')
            
            timestamps.append({
                "chunk_id": i,
                "start": str(start),
                "end": str(end),
                "text_preview": chunk.get('text', '')[:50] + "..." if len(chunk.get('text', '')) > 50 else chunk.get('text', '')
            })
    
    return timestamps

def recommend_ad_placements(transcript: List[Dict], timestamps: List[Dict]) -> List[Dict[str, Any]]:
    """Recommend ad placement based on content analysis"""
    ad_recommendations = []
    
    # Look for natural breaks in content
    for i, chunk in enumerate(transcript):
        text = chunk.get('text', '').lower()
        
        # Skip if chunk is too short
        if len(text.split()) < 5:
            continue
        
        # Look for scene transitions
        scene_indicators = ["cut to", "fade in", "fade out", "int.", "ext.", "later", "meanwhile", "next day"]
        is_scene_break = any(indicator in text for indicator in scene_indicators)
        
        # Avoid dialogue-heavy scenes
        dialogue_indicators = ['"', "'", "said", "asked", "replied"]
        dialogue_count = sum(text.count(indicator) for indicator in dialogue_indicators)
        is_dialogue_heavy = dialogue_count > len(text.split()) * 0.3
        
        # Avoid action scenes
        action_words = ["fight", "chase", "explosion", "battle", "attack", "running"]
        is_action_scene = any(word in text for word in action_words)
        
        # Check for emotional intensity
        emotional_words = ["crying", "screaming", "terrified", "passionate", "intense"]
        is_emotional = any(word in text for word in emotional_words)
        
        # Calculate suitability score
        suitability_score = 0.5
        reasons = []
        
        if is_scene_break:
            suitability_score += 0.3
            reasons.append("Scene transition detected")
        
        if not is_dialogue_heavy:
            suitability_score += 0.2
        else:
            suitability_score -= 0.2
            reasons.append("Heavy dialogue")
        
        if not is_action_scene:
            suitability_score += 0.2
        else:
            suitability_score -= 0.3
            reasons.append("Action sequence")
        
        if not is_emotional:
            suitability_score += 0.1
        else:
            suitability_score -= 0.2
            reasons.append("Emotional scene")
        
        # Only recommend if score is above threshold
        if suitability_score > 0.6:
            timestamp_info = next((ts for ts in timestamps if ts['chunk_id'] == i), None)
            
            ad_recommendations.append({
                "chunk_id": i,
                "timestamp": timestamp_info['end'] if timestamp_info else f"chunk_{i}_end",
                "suitability_score": round(suitability_score, 2),
                "placement_type": "scene_break" if is_scene_break else "natural_pause",
                "reasons": reasons[:2],  # Top 2 reasons
                "text_context": chunk.get('text', '')[:100] + "..." if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
            })
    
    # Sort by suitability score and return top recommendations
    ad_recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)
    return ad_recommendations[:10]  # Top 10 ad placement recommendations

def analyze_script(transcript_data: Dict) -> Dict[str, Any]:
    """Perform complete script-level analysis"""
    print("Starting comprehensive script analysis...")
    
    transcript = transcript_data.get('transcript', [])
    if not transcript:
        print("No transcript data found")
        return {}
    
    start_time = time.time()
    
    # Extract all text for analysis
    full_text = extract_all_text(transcript)
    print(f"Analyzing script with {len(transcript)} chunks and {len(full_text.split())} total words")
    
    # Perform analyses
    print("Extracting keywords...")
    top_keywords = extract_top_keywords(full_text, 10)
    
    print("Extracting named entities...")
    all_entities = extract_all_entities(transcript)
    
    print("Identifying lead characters...")
    lead_characters = identify_lead_characters(all_entities, 10)
    
    print("Analyzing sentiment...")
    overall_sentiment = analyze_overall_sentiment(transcript)
    
    print("Classifying content...")
    content_classification = classify_content(full_text)
    
    print("Detecting sensitive content...")
    sensitive_analysis = detect_sensitive_content_overall(full_text)
    
    print("Extracting timestamps...")
    timestamps = extract_timestamps(transcript)
    
    print("Generating ad placement recommendations...")
    ad_recommendations = recommend_ad_placements(transcript, timestamps)
    
    processing_time = time.time() - start_time
    
    # Compile final analysis
    analysis = {
        "script_metadata": {
            "total_chunks": len(transcript),
            "total_words": len(full_text.split()),
            "processing_time_seconds": round(processing_time, 2),
            "analysis_timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        },
        
        "top_keywords": top_keywords,
        
        "character_analysis": {
            "lead_characters": lead_characters,
            "total_unique_characters": len(all_entities.get("PER", {})),
            "character_mentions": all_entities.get("PER", {})
        },
        
        "location_analysis": {
            "primary_locations": [
                {"name": loc, "mentions": count}
                for loc, count in list(all_entities.get("LOC", {}).items())[:10]
            ],
            "total_unique_locations": len(all_entities.get("LOC", {}))
        },
        
        "organization_analysis": {
            "organizations": [
                {"name": org, "mentions": count}
                for org, count in list(all_entities.get("ORG", {}).items())[:10]
            ],
            "total_unique_organizations": len(all_entities.get("ORG", {}))
        },
        
        "named_entity_summary": {
            "people": len(all_entities.get("PER", {})),
            "locations": len(all_entities.get("LOC", {})),
            "organizations": len(all_entities.get("ORG", {})),
            "total_entities": sum(len(entities) for entities in all_entities.values())
        },
        
        "sentiment_analysis": overall_sentiment,
        
        "content_classification": {
            "primary_genres": content_classification,
            "content_rating_indicators": sensitive_analysis
        },
        
        "timeline_analysis": {
            "total_timestamps": len(timestamps),
            "timestamp_coverage": "full" if timestamps else "none",
            "timeline_data": timestamps[:20]  # First 20 timestamps as sample
        },
        
        "ad_placement_recommendations": {
            "recommended_slots": len(ad_recommendations),
            "top_recommendations": ad_recommendations,
            "placement_strategy": "scene_transition_based"
        }
    }
    
    print(f"Analysis completed in {processing_time:.2f} seconds")
    return analysis

def process_script_file(file_path: Path) -> Dict[str, Any]:
    """Process a single script file"""
    print(f"Processing: {file_path.name}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if 'transcript' not in data:
            print(f"Invalid JSON structure - missing 'transcript' key")
            return None
        
        analysis = analyze_script(data)
        
        if analysis:
            analysis["source_file"] = file_path.name
            return analysis
        
        return None
        
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None

def process_all_scripts():
    """Process all script files and generate overall analysis"""
    input_dir = Path("data/processed/NLP_jsons")
    output_dir = Path("data/processed/processed_nlp_validator_jsons")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 0
    
    print(f"Found {len(json_files)} script files to analyze")
    
    successful = 0
    failed = 0
    
    for file_path in json_files:
        result = process_script_file(file_path)
        
        if result:
            # Save individual analysis
            output_file = output_dir / f"{file_path.stem}_analysis.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            print(f"Saved analysis: {output_file}")
            successful += 1
        else:
            failed += 1
    
    print(f"✅ NLP Processing Summary: {successful} successful, {failed} failed")
    return successful

def check_environment():
    """Check environment setup"""
    print("Environment Check:")
    issues = []
    
    if nlp:
        print("- SpaCy model available")
    else:
        print("- SpaCy model missing")
        issues.append("spacy")
    
    if vader_analyzer:
        print("- VADER sentiment analyzer available")
    else:
        print("- VADER not available, using TextBlob fallback")
    
    try:
        TextBlob("test")
        print("- TextBlob available")
    except:
        print("- TextBlob missing")
        issues.append("textblob")
    
    return len(issues) == 0

# Main execution
if __name__ == "__main__":
    print("Script-Level NLP Analysis Tool")
    print("Hybrid approach: Fast local processing with comprehensive analysis")
    print("=" * 60)
    
    if not check_environment():
        print("Please install missing dependencies:")
        print("pip install spacy textblob vaderSentiment requests python-dotenv")
        print("python -m spacy download en_core_web_sm")
        exit(1)
    
    print("Environment ready! Starting script analysis...")
    
    start_time = time.time()
    process_all_scripts()
    total_time = time.time() - start_time
    
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print("Script analysis complete!")