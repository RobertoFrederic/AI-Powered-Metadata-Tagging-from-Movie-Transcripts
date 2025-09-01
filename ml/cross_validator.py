import json
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter
import math

class CrossValidator:
    def __init__(self):
        self.llm_data = None
        self.nlp_data = None
        self.comparison_results = {}
    
    def load_analysis_files(self, llm_file_path: str, nlp_file_path: str) -> bool:
        """Load both analysis files for comparison"""
        try:
            with open(llm_file_path, 'r', encoding='utf-8') as f:
                self.llm_data = json.load(f)
            with open(nlp_file_path, 'r', encoding='utf-8') as f:
                self.nlp_data = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading files: {e}")
            return False
    def set_processing_info(self, processing_info: Dict):
        """Set processing timing information from the pipeline"""
        self.processing_info = processing_info
    
    def calculate_overall_confidence_score(self) -> Dict[str, Any]:
        """Calculate overall confidence comparison between LLM and NLP"""
        llm_confidence = self._calculate_llm_confidence()
        nlp_confidence = self._calculate_nlp_confidence()
        agreement_score = self._calculate_agreement_score()
        
        return {
            "llm_confidence": round(llm_confidence, 2),
            "nlp_confidence": round(nlp_confidence, 2),
            "agreement_score": round(agreement_score, 2),
            "overall_confidence": round((llm_confidence + nlp_confidence + agreement_score) / 3, 2),
            "venn_data": self._generate_venn_diagram_data()
        }
    
    def _calculate_llm_confidence(self) -> float:
        """Calculate average confidence from LLM analysis"""
        if not self.llm_data:
            return 0.0
        
        confidences = []
        
        # Genre confidence
        if 'content_classification' in self.llm_data:
            for genre in self.llm_data['content_classification']['primary_genres']:
                confidences.append(genre.get('confidence', 0))
        
        # Character emotion confidence
        if 'lead_characters' in self.llm_data:
            for char in self.llm_data['lead_characters']:
                confidences.append(char.get('emotion_confidence', 0))
        
        # Sentiment confidence
        if 'overall_sentiment' in self.llm_data:
            confidences.append(self.llm_data['overall_sentiment'].get('confidence', 0))
        
        return np.mean(confidences) if confidences else 0.0
    
    def _calculate_nlp_confidence(self) -> float:
        """Calculate average confidence from NLP analysis"""
        if not self.nlp_data:
            return 0.0
        
        confidences = []
        
        # Sentiment confidence
        if 'sentiment_analysis' in self.nlp_data:
            confidences.append(self.nlp_data['sentiment_analysis'].get('confidence', 0))
        
        # Content classification confidence
        if 'content_classification' in self.nlp_data:
            content_warnings = self.nlp_data['content_classification']['content_rating_indicators']
            for category, score in content_warnings.get('confidence_scores', {}).items():
                confidences.append(score)
        
        return np.mean(confidences) if confidences else 0.0
    
    def _calculate_agreement_score(self) -> float:
        """Calculate agreement between LLM and NLP results"""
        agreement_scores = []
        
        # Compare sentiment analysis
        sentiment_agreement = self._compare_sentiment()
        agreement_scores.append(sentiment_agreement)
        
        # Compare character extraction
        character_agreement = self._compare_characters()
        agreement_scores.append(character_agreement)
        
        # Compare keyword extraction
        keyword_agreement = self._compare_keywords()
        agreement_scores.append(keyword_agreement)
        
        return np.mean(agreement_scores) if agreement_scores else 0.0
    
    def _compare_sentiment(self) -> float:
        """Compare sentiment analysis between LLM and NLP"""
        if not (self.llm_data and self.nlp_data):
            return 0.0
        
        llm_sentiment = self.llm_data.get('overall_sentiment', {}).get('classification', '')
        nlp_sentiment = self.nlp_data.get('sentiment_analysis', {}).get('overall_sentiment', '')
        
        return 1.0 if llm_sentiment.lower() == nlp_sentiment.lower() else 0.5
    
    def _compare_characters(self) -> float:
        """Compare character extraction accuracy"""
        if not (self.llm_data and self.nlp_data):
            return 0.0
        
        llm_chars = set()
        if 'lead_characters' in self.llm_data:
            llm_chars = {char['name'].lower() for char in self.llm_data['lead_characters']}
        
        nlp_chars = set()
        if 'character_analysis' in self.nlp_data:
            nlp_chars = {char['name'].lower() for char in self.nlp_data['character_analysis']['lead_characters']}
        
        if not llm_chars and not nlp_chars:
            return 1.0
        if not llm_chars or not nlp_chars:
            return 0.0
        
        intersection = len(llm_chars.intersection(nlp_chars))
        union = len(llm_chars.union(nlp_chars))
        
        return intersection / union if union > 0 else 0.0
    
    def _compare_keywords(self) -> float:
        """Compare keyword extraction overlap"""
        if not (self.llm_data and self.nlp_data):
            return 0.0
        
        llm_keywords = set()
        if 'top_keywords' in self.llm_data:
            llm_keywords = {kw['keyword'].lower() for kw in self.llm_data['top_keywords'][:10]}
        
        nlp_keywords = set()
        if 'top_keywords' in self.nlp_data:
            nlp_keywords = {kw['keyword'].lower() for kw in self.nlp_data['top_keywords'][:10]}
        
        if not llm_keywords and not nlp_keywords:
            return 1.0
        if not llm_keywords or not nlp_keywords:
            return 0.0
        
        intersection = len(llm_keywords.intersection(nlp_keywords))
        union = len(llm_keywords.union(nlp_keywords))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_venn_diagram_data(self) -> Dict:
        """Generate Venn diagram data showing LLM vs NLP overlap"""
    
    # Calculate based on actual keyword overlap
        if self.llm_data and self.nlp_data:
            llm_kw = set()
            nlp_kw = set()
            
            # Extract LLM keywords
            if 'top_keywords' in self.llm_data:
                llm_kw = {kw['keyword'].lower() for kw in self.llm_data['top_keywords'][:10]}
            
            # Extract NLP keywords  
            if 'top_keywords' in self.nlp_data:
                nlp_kw = {kw['keyword'].lower() for kw in self.nlp_data['top_keywords'][:10]}
            
            both = len(llm_kw.intersection(nlp_kw))
            llm_only = len(llm_kw - nlp_kw)
            nlp_only = len(nlp_kw - llm_kw)
            
            # Only use defaults if no real data found
            if both == 0 and llm_only == 0 and nlp_only == 0:
                llm_only, nlp_only, both = 10, 10, 0  # More realistic defaults
            
            return {
                "llm_only": llm_only,
                "nlp_only": nlp_only,
                "both": both,
                "total": llm_only + nlp_only + both
            }
        
        # Fallback defaults only if no data loaded
        return {"llm_only": 10, "nlp_only": 10, "both": 0, "total": 20}
    
    def generate_performance_metrics(self) -> Dict[str, Any]:
        """Generate performance comparison metrics"""
        return {
            "processing_speed": self._compare_processing_speed(),
            "accuracy_metrics": self._calculate_accuracy_metrics(),
            "coverage_analysis": self._analyze_coverage(),
            "triangle_chart_data": self._generate_triangle_chart_data()
        }
    
    def _compare_processing_speed(self) -> Dict:
        """Compare processing speeds using real data"""
        llm_time = 20  # Default LLM time
        nlp_time = 10  # Default NLP time
        
        # Try to get real processing times from your API data
        if hasattr(self, 'processing_info'):
            nlp_time = getattr(self.processing_info, 'nlp_processing_time', 10)
            # LLM time estimation based on complexity
            if self.llm_data and 'script_metadata' in self.llm_data:
                word_count = self.llm_data['script_metadata'].get('total_words', 0)
                llm_time = max(15, word_count / 1000)  # Estimate based on word count
        
        return {
            "llm_time": round(llm_time, 1),
            "nlp_time": round(nlp_time, 1),
            "speed_ratio": round(nlp_time / llm_time if llm_time > 0 else 1, 2)
        }
    
    def _calculate_accuracy_metrics(self) -> Dict:
        """Calculate accuracy comparison metrics"""
        return {
            "sentiment_accuracy": {
                "llm": round(self._get_llm_sentiment_accuracy(), 2),
                "nlp": round(self._get_nlp_sentiment_accuracy(), 2)
            },
            "entity_extraction_accuracy": {
                "llm": round(self._get_llm_entity_accuracy(), 2),
                "nlp": round(self._get_nlp_entity_accuracy(), 2)
            },
            "keyword_relevance": {
                "llm": round(self._get_llm_keyword_relevance(), 2),
                "nlp": round(self._get_nlp_keyword_relevance(), 2)
            }
        }
    
    def _get_llm_sentiment_accuracy(self) -> float:
        """Estimate LLM sentiment analysis accuracy"""
        if self.llm_data and 'overall_sentiment' in self.llm_data:
            return self.llm_data['overall_sentiment'].get('confidence', 0.7) * 100
        return 70.0
    
    def _get_nlp_sentiment_accuracy(self) -> float:
        """Get NLP sentiment analysis accuracy"""
        if self.nlp_data and 'sentiment_analysis' in self.nlp_data:
            return self.nlp_data['sentiment_analysis'].get('confidence', 0.5) * 100
        return 50.0
    
    def _get_llm_entity_accuracy(self) -> float:
        """Estimate LLM entity extraction accuracy"""
        if self.llm_data and 'named_entity_summary' in self.llm_data:
            total_entities = self.llm_data['named_entity_summary'].get('person_count', 0)
            return min(85.0, 70 + (total_entities * 2))  # Heuristic based on entity count
        return 75.0
    
    def _get_nlp_entity_accuracy(self) -> float:
        """Calculate NLP entity extraction accuracy"""
        if self.nlp_data and 'named_entity_summary' in self.nlp_data:
            total_entities = self.nlp_data['named_entity_summary'].get('total_entities', 0)
            return min(95.0, 70 + (total_entities * 1.5))  # Heuristic based on entity count
        return 80.0
    
    def _get_llm_keyword_relevance(self) -> float:
        """Calculate LLM keyword relevance score"""
        if self.llm_data and 'top_keywords' in self.llm_data:
            avg_percentage = np.mean([kw.get('percentage', 0) for kw in self.llm_data['top_keywords'][:5]])
            return avg_percentage * 100  # Convert to readable percentage
        return 65.0
    
    def _get_nlp_keyword_relevance(self) -> float:
        """Calculate NLP keyword relevance score"""
        if self.nlp_data and 'top_keywords' in self.nlp_data:
            avg_relevance = np.mean([kw.get('relevance', 0) for kw in self.nlp_data['top_keywords'][:5]])
            return avg_relevance * 100  # Convert to readable percentage
        return 70.0
    
    def _analyze_coverage(self) -> Dict:
        """Analyze coverage comparison between methods"""
        llm_coverage = 0
        nlp_coverage = 0
        
        if self.llm_data:
            llm_coverage = len(self.llm_data.get('lead_characters', []))
        
        if self.nlp_data:
            nlp_coverage = self.nlp_data.get('named_entity_summary', {}).get('total_entities', 0)
        
        return {
            "llm_coverage": llm_coverage,
            "nlp_coverage": nlp_coverage,
            "coverage_ratio": round(llm_coverage / nlp_coverage if nlp_coverage > 0 else 1, 2)
        }
    
    def _generate_triangle_chart_data(self) -> Dict:
        """Generate triangle chart data for comparative metrics"""
        llm_metrics = self._get_llm_performance_metrics()
        nlp_metrics = self._get_nlp_performance_metrics()
        
        return {
            "llm_metrics": llm_metrics,
            "nlp_metrics": nlp_metrics,
            "categories": ["Accuracy", "Speed", "Coverage"]
        }
    
    def _get_llm_performance_metrics(self) -> List[float]:
        """Get LLM performance metrics for triangle chart"""
        accuracy = self._get_llm_sentiment_accuracy()
        speed = 60.0  # Lower is better for speed, so inverse score
        coverage = self._analyze_coverage()['llm_coverage'] * 5  # Scale for visualization
        
        return [accuracy, speed, min(coverage, 100)]
    
    def _get_nlp_performance_metrics(self) -> List[float]:
        """Get NLP performance metrics for triangle chart"""
        accuracy = self._get_nlp_sentiment_accuracy()
        speed = 85.0  # Higher speed score for faster processing
        coverage = self._analyze_coverage()['nlp_coverage'] * 2  # Scale for visualization
        
        return [accuracy, speed, min(coverage, 100)]
    
    def generate_accuracy_comparison(self) -> Dict[str, Any]:
        """Generate detailed accuracy comparison data"""
        return {
            "sentiment_comparison": self._detailed_sentiment_comparison(),
            "entity_comparison": self._detailed_entity_comparison(),
            "keyword_comparison": self._detailed_keyword_comparison(),
            "overall_accuracy_scores": {
                "llm": round((self._get_llm_sentiment_accuracy() + self._get_llm_entity_accuracy()) / 2, 1),
                "nlp": round((self._get_nlp_sentiment_accuracy() + self._get_nlp_entity_accuracy()) / 2, 1)
            }
        }
    
    def _detailed_sentiment_comparison(self) -> Dict:
        """Detailed sentiment analysis comparison"""
        llm_sentiment = self.llm_data.get('overall_sentiment', {}) if self.llm_data else {}
        nlp_sentiment = self.nlp_data.get('sentiment_analysis', {}) if self.nlp_data else {}
        
        return {
            "llm_classification": llm_sentiment.get('classification', 'unknown'),
            "nlp_classification": nlp_sentiment.get('overall_sentiment', 'unknown'),
            "llm_confidence": round(llm_sentiment.get('confidence', 0) * 100, 1),
            "nlp_confidence": round(nlp_sentiment.get('confidence', 0) * 100, 1),
            "agreement": llm_sentiment.get('classification', '').lower() == nlp_sentiment.get('overall_sentiment', '').lower()
        }
    
    def _detailed_entity_comparison(self) -> Dict:
        """Detailed entity extraction comparison"""
        llm_entities = len(self.llm_data.get('lead_characters', [])) if self.llm_data else 0
        nlp_entities = self.nlp_data.get('named_entity_summary', {}).get('total_entities', 0) if self.nlp_data else 0
        
        return {
            "llm_total_entities": llm_entities,
            "nlp_total_entities": nlp_entities,
            "entity_overlap": self._calculate_entity_overlap(),
            "precision_score": round(self._calculate_entity_overlap(), 2)
        }
    
    def _detailed_keyword_comparison(self) -> Dict:
        """Detailed keyword extraction comparison"""
        llm_keywords = self.llm_data.get('top_keywords', [])[:10] if self.llm_data else []
        nlp_keywords = self.nlp_data.get('top_keywords', [])[:10] if self.nlp_data else []
        
        overlap = self._calculate_keyword_overlap(llm_keywords, nlp_keywords)
        
        return {
            "llm_keyword_count": len(llm_keywords),
            "nlp_keyword_count": len(nlp_keywords),
            "keyword_overlap_percentage": round(overlap * 100, 1),
            "top_shared_keywords": self._get_shared_keywords(llm_keywords, nlp_keywords)
        }
    
    def _calculate_entity_overlap(self) -> float:
        """Calculate entity overlap between LLM and NLP"""
        if not (self.llm_data and self.nlp_data):
            return 0.0
        
        llm_chars = {char['name'].lower() for char in self.llm_data.get('lead_characters', [])}
        nlp_chars = {char['name'].lower() for char in self.nlp_data.get('character_analysis', {}).get('lead_characters', [])}
        
        if not llm_chars or not nlp_chars:
            return 0.0
        
        intersection = len(llm_chars.intersection(nlp_chars))
        union = len(llm_chars.union(nlp_chars))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_keyword_overlap(self, llm_kw: List, nlp_kw: List) -> float:
        """Calculate keyword overlap percentage"""
        llm_set = {kw['keyword'].lower() for kw in llm_kw}
        nlp_set = {kw['keyword'].lower() for kw in nlp_kw}
        
        if not llm_set or not nlp_set:
            return 0.0
        
        intersection = len(llm_set.intersection(nlp_set))
        union = len(llm_set.union(nlp_set))
        
        return intersection / union if union > 0 else 0.0
    
    def _get_shared_keywords(self, llm_kw: List, nlp_kw: List) -> List[str]:
        """Get top shared keywords between LLM and NLP"""
        llm_set = {kw['keyword'].lower() for kw in llm_kw}
        nlp_set = {kw['keyword'].lower() for kw in nlp_kw}
        
        shared = list(llm_set.intersection(nlp_set))
        return shared[:5]  # Return top 5 shared keywords
    
    def generate_cross_validation_report(self) -> Dict[str, Any]:
        """Generate complete cross-validation analysis for confidence tab"""
        return {
            "confidence_scores": self.calculate_overall_confidence_score(),
            "performance_metrics": self.generate_performance_metrics(),
            "accuracy_comparison": self.generate_accuracy_comparison(),
            "validation_summary": self._generate_validation_summary()
        }
    
    def _generate_validation_summary(self) -> Dict:
        """Generate validation summary statistics"""
        llm_conf = self._calculate_llm_confidence()
        nlp_conf = self._calculate_nlp_confidence()  
        agreement = self._calculate_agreement_score()
        
        # Use weighted calculation for more realistic reliability
        overall_reliability = (llm_conf * 0.4 + nlp_conf * 0.3 + agreement * 0.3) * 100
        
        return {
            "validation_status": "high" if agreement > 0.7 else "medium" if agreement > 0.4 else "low",
            "recommendation": self._get_validation_recommendation(agreement),
            "confidence_delta": round(abs(llm_conf - nlp_conf), 2),
            "overall_reliability": round(overall_reliability, 1)
        }
    
    def _get_validation_recommendation(self, agreement_score: float) -> str:
        """Get recommendation based on validation results"""
        if agreement_score > 0.7:
            return "High agreement - Results are reliable for production use"
        elif agreement_score > 0.4:
            return "Medium agreement - Consider additional validation"
        else:
            return "Low agreement - Requires manual review and adjustment"
# Add this function at the end of your cross_validator.py file, before the if __name__ == "__main__": block

def create_cross_validation_data(llm_file_path: str = None, nlp_file_path: str = None) -> Dict[str, Any]:
    """
    Main function to create cross-validation data from analysis files
    If no file paths provided, will auto-detect latest files
    """
    import os
    import glob
    
    validator = CrossValidator()
    
    # Auto-detect files if not provided
    if not llm_file_path or not nlp_file_path:
        llm_dir = os.path.join("data", "processed", "processed_llm_analyzer_jsons")
        nlp_dir = os.path.join("data", "processed", "processed_nlp_validator_jsons")
        
        def get_latest_file(folder):
            """Return the newest JSON file in the given folder."""
            files = glob.glob(os.path.join(folder, "*.json"))
            if not files:
                return None
            return max(files, key=os.path.getctime)
        
        llm_file_path = get_latest_file(llm_dir)
        nlp_file_path = get_latest_file(nlp_dir)
        
        if not llm_file_path or not nlp_file_path:
            return {"error": "Could not find analysis files for cross-validation"}
    
    if not validator.load_analysis_files(llm_file_path, nlp_file_path):
        return {"error": "Failed to load analysis files"}
    
    return validator.generate_cross_validation_report()
# Add this to the end of cross_validator.py

# Replace the _main_ section at the bottom of cross_validator.py
if __name__ == "__main__":
    import os
    import glob
    import json

    # Base folders for processed files
    llm_dir = os.path.join("data", "processed", "processed_llm_analyzer_jsons")
    nlp_dir = os.path.join("data", "processed", "processed_nlp_validator_jsons")

    def get_latest_file(folder):
        """Return the newest JSON file in the given folder."""
        files = glob.glob(os.path.join(folder, "*.json"))
        if not files:
            return None
        return max(files, key=os.path.getctime)

    # Automatically pick the latest LLM + NLP files
    llm_file = get_latest_file(llm_dir)
    nlp_file = get_latest_file(nlp_dir)

    print(f"Looking for LLM file: {llm_file}")
    print(f"Looking for NLP file: {nlp_file}")

    # Check if files exist
    if not llm_file or not os.path.exists(llm_file):
        print(f"❌ LLM file not found in {llm_dir}")
        if os.path.exists(llm_dir):
            print("Available files:")
            for file in os.listdir(llm_dir):
                if file.endswith(".json"):
                    print(f"  - {file}")
        exit(1)

    if not nlp_file or not os.path.exists(nlp_file):
        print(f"❌ NLP file not found in {nlp_dir}")
        if os.path.exists(nlp_dir):
            print("Available files:")
            for file in os.listdir(nlp_dir):
                if file.endswith(".json"):
                    print(f"  - {file}")
        exit(1)

    # Create validator and generate data
    validator = CrossValidator()
    if validator.load_analysis_files(llm_file, nlp_file):
        cross_val_data = validator.generate_cross_validation_report()

        # Save to data/processed/cross_validation_output.json
        output_path = os.path.join("data", "processed","cross_validator", "cross_validation_output.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cross_val_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Cross-validation data saved to: {output_path}")
    else:
        print("❌ Failed to load files")