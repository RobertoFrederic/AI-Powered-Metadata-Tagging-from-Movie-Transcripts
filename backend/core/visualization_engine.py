import json
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path

class VisualizationEngine:
    def __init__(self):
        self.llm_data = None
        self.nlp_data = None
    
    def load_analysis_files(self, llm_file_path: str, nlp_file_path: str) -> bool:
        """Load both LLM and NLP analysis files"""
        try:
            with open(llm_file_path, 'r', encoding='utf-8') as f:
                self.llm_data = json.load(f)
            with open(nlp_file_path, 'r', encoding='utf-8') as f:
                self.nlp_data = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading files: {e}")
            return False
    
    def generate_metadata_tab_data(self) -> Dict[str, Any]:
        """Generate all data for metadata tab visualizations"""
        return {
            "lead_characters": self._get_lead_characters_data(),
            "genre_classification": self._get_genre_data(),
            "content_classification_plot": self._get_content_classification_plot(),
            "keywords_plot": self._get_keywords_plot(),
            "sentiment_pie": self._get_sentiment_pie_data(),
            "emotion_pie": self._get_emotion_pie_data(),
            "named_entities": self._get_named_entities_data(),
            "emotion_radar": self._get_emotion_radar_data(),
            "synopsis_summary": self._get_synopsis_summary()
        }
    
    def _get_lead_characters_data(self) -> List[Dict]:
        """Extract lead characters with importance percentages"""
        characters = []
        if self.llm_data and 'lead_characters' in self.llm_data:
            for char in self.llm_data['lead_characters'][:7]:
                characters.append({
                    "name": char['name'],
                    "importance": char.get('importance_percentage', 0),
                    "mentions": char.get('total_mentions', 0),
                    "emotion": char.get('dominant_emotion', 'neutral')
                })
        return characters
    
    def _get_genre_data(self) -> List[Dict]:
        """Extract genre classification data"""
        genres = []
        if self.llm_data and 'content_classification' in self.llm_data:
            for genre in self.llm_data['content_classification']['primary_genres']:
                genres.append({
                    "genre": genre['genre'],
                    "confidence": round(genre['confidence'] * 100, 1)
                })
        return genres
    
    def _get_content_classification_plot(self) -> Dict:
        """Prepare content classification bar plot data"""
        if not self.llm_data or 'content_classification' not in self.llm_data:
            return {"categories": [], "scores": []}
        
        categories = []
        scores = []
        for genre in self.llm_data['content_classification']['primary_genres'][:5]:
            categories.append(genre['genre'].title())   # fixed key
            scores.append(round(genre['confidence'] * 100, 1))
        
        return {"categories": categories, "scores": scores}

    
    def _get_keywords_plot(self) -> Dict:
        """Prepare keywords bar plot with percentages"""
        if not self.llm_data or 'top_keywords' not in self.llm_data:
            return {"keywords": [], "percentages": []}
        
        keywords = []
        percentages = []
        for kw in self.llm_data['top_keywords'][:10]:
            keywords.append(kw['keyword'].title())
            percentages.append(round(kw['percentage'] * 100, 2))
        
        return {"keywords": keywords, "percentages": percentages}
    
    def _get_sentiment_pie_data(self) -> Dict:
        """Extract sentiment distribution for pie chart"""
        if self.nlp_data and 'sentiment_analysis' in self.nlp_data:
            dist = self.nlp_data['sentiment_analysis']['distribution']
            return {
                "labels": ["Positive", "Negative", "Neutral"],
                "values": [
                    round(dist.get('positive', 0) * 100, 1),
                    round(dist.get('negative', 0) * 100, 1),
                    round(dist.get('neutral', 0) * 100, 1)
                ]
            }
        return {"labels": [], "values": []}
    
    def _get_emotion_pie_data(self) -> Dict:
        """Extract emotion distribution from characters"""
        emotions = {}
        if self.llm_data and 'lead_characters' in self.llm_data:
            for char in self.llm_data['lead_characters']:
                emotion = char.get('dominant_emotion', 'neutral')
                emotions[emotion] = emotions.get(emotion, 0) + 1
        
        return {
            "labels": list(emotions.keys()),
            "values": list(emotions.values())
        }
    
    def _get_named_entities_data(self) -> Dict:
        """Extract named entity recognition data with top names"""
        if self.llm_data and 'named_entity_summary' in self.llm_data:
            nes = self.llm_data['named_entity_summary']
            top = nes.get('top_entities', {})
            return {
                "people_count": nes.get('person_count', 0),
                "people": top.get('persons', []),
                "locations_count": nes.get('location_count', 0),
                "locations": top.get('locations', []),
                "organizations_count": nes.get('organization_count', 0),
                "organizations": top.get('organizations', []),
                "total": nes.get('person_count', 0) + nes.get('location_count', 0) + nes.get('organization_count', 0)
            }
        return {
            "people_count": 0, "people": [],
            "locations_count": 0, "locations": [],
            "organizations_count": 0, "organizations": [],
            "total": 0
        }

    
    def _get_emotion_radar_data(self) -> Dict:
        """Prepare emotion radar chart data for main characters"""
        emotions = ['neutral', 'joy', 'anger', 'fear', 'sadness']
        emotion_scores = {emotion: 0 for emotion in emotions}
        
        if self.llm_data and 'lead_characters' in self.llm_data:
            for char in self.llm_data['lead_characters'][:5]:
                emotion = char.get('dominant_emotion', 'neutral')
                confidence = char.get('emotion_confidence', 0)
                if emotion in emotion_scores:
                    emotion_scores[emotion] += confidence
        
        return {
            "labels": emotions,
            "values": [round(emotion_scores[e] * 100, 1) for e in emotions]
        }
    
    def _get_synopsis_summary(self) -> Dict:
        """Get synopsis and summary text"""
        synopsis = ""
        if self.llm_data and 'script_synopsis' in self.llm_data:
            synopsis = self.llm_data['script_synopsis']
        
        return {
            "synopsis": synopsis,
            "word_count": self.llm_data.get('script_metadata', {}).get('total_words', 0) if self.llm_data else 0,
            "duration": self.llm_data.get('script_metadata', {}).get('estimated_duration_minutes', 0) if self.llm_data else 0
        }
    
    def generate_ad_insights_tab_data(self) -> Dict[str, Any]:
        """Generate all data for ad insights tab"""
        return {
            "ad_placement_timeline": self._get_ad_placement_timeline(),
            "ad_recommendations": self._get_ad_recommendations_data(),
            "placement_strategy": self._get_placement_strategy_data()
        }
    
    def _get_ad_placement_timeline(self) -> Dict:
        """Prepare ad placement timeline graph data"""
        timeline_data = []
        
        if self.llm_data and 'ad_placement_recommendations' in self.llm_data:
            for i, placement in enumerate(self.llm_data['ad_placement_recommendations']['optimal_placements']):
                timeline_data.append({
                    "id": placement['placement_id'],
                    "timestamp": placement.get('timestamp_estimate', f"00:{i*20:02d}:00"),
                    "suitability": round(placement['suitability_score'] * 100, 1),
                    "scene": placement['scene_context'][:50] + "..."
                })
        
        return {
            "placements": timeline_data,
            "total_slots": len(timeline_data)
        }
    
    def _get_ad_recommendations_data(self) -> List[Dict]:
        """Extract ad recommendations with details"""
        recommendations = []
        
        if self.llm_data and 'ad_placement_recommendations' in self.llm_data:
            for placement in self.llm_data['ad_placement_recommendations']['optimal_placements']:
                recommendations.append({
                    "placement_id": placement['placement_id'],
                    "scene": placement['scene_context'],
                    "ad_types": placement['recommended_ad_types'],
                    "suitability": round(placement['suitability_score'] * 100, 1),
                    "reasoning": placement['reasoning']
                })
        
        return recommendations
    
    def _get_placement_strategy_data(self) -> Dict:
        """Get placement strategy overview"""
        strategy = "scene_transition_based"
        total_slots = 0
        
        if self.nlp_data and 'ad_placement_recommendations' in self.nlp_data:
            strategy = self.nlp_data['ad_placement_recommendations'].get('placement_strategy', strategy)
            total_slots = self.nlp_data['ad_placement_recommendations'].get('recommended_slots', 0)
        
        return {
            "strategy": strategy,
            "total_recommended_slots": total_slots,
            "average_suitability": 85.2  # Calculate from actual data
        }
    
    def generate_complete_visualization_data(self) -> Dict[str, Any]:
        """Generate complete visualization data for all tabs"""
        return {
            "metadata_tab": self.generate_metadata_tab_data(),
            "ad_insights_tab": self.generate_ad_insights_tab_data(),
            "processing_info": {
                "llm_processing_time": self.llm_data.get('script_metadata', {}).get('analysis_timestamp', '') if self.llm_data else '',
                "nlp_processing_time": self.nlp_data.get('script_metadata', {}).get('processing_time_seconds', 0) if self.nlp_data else 0,
                "total_chunks": self.nlp_data.get('script_metadata', {}).get('total_chunks', 0) if self.nlp_data else 0
            }
        }

def create_visualization_data(llm_file_path: str, nlp_file_path: str) -> Dict[str, Any]:
    """Main function to create visualization data from analysis files"""
    engine = VisualizationEngine()
    
    if not engine.load_analysis_files(llm_file_path, nlp_file_path):
        return {"error": "Failed to load analysis files"}
    
    return engine.generate_complete_visualization_data()

def generate_latest_visualization():
    """Generate visualization data from latest processed files"""
    llm_dir = "data/processed/processed_llm_analyzer_jsons"
    nlp_dir = "data/processed/processed_nlp_validator_jsons"
    
    import glob
    
    def get_latest_file(folder):
        files = glob.glob(os.path.join(folder, "*.json"))
        if not files:
            return None
        return max(files, key=os.path.getctime)
    
    llm_file = get_latest_file(llm_dir)
    nlp_file = get_latest_file(nlp_dir)
    
    if llm_file and nlp_file:
        engine = VisualizationEngine()
        if engine.load_analysis_files(llm_file, nlp_file):
            viz_data = engine.generate_complete_visualization_data()
            
            output_path = "data/processed/visualization_engine/visualization_data.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(viz_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Visualization data saved")
            return True
    
    return False