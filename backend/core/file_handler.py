import os
import re
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import chardet
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileHandler:
    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            current_dir = Path(__file__).parent
            while current_dir.parent != current_dir:
                if (current_dir / 'main.py').exists():
                    project_root = current_dir
                    break
                current_dir = current_dir.parent
            else:
                project_root = Path(__file__).parent.parent.parent
        
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.uploads_dir = self.data_dir / "uploads"
        self.processed_dir = self.data_dir / "processed"
        
        for directory in [self.data_dir, self.uploads_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.supported_formats = {'.txt', '.srt', '.vtt', '.json'}
        
        # Token limits for different models
        self.model_limits = {
            'gpt-4': 128000,
            'gpt-4-turbo': 128000,
            'claude-3': 200000,
            'claude-sonnet-4': 200000
        }
        
        # Enhanced timestamp patterns
        self.timestamp_patterns = [
            # SRT/VTT format: 00:01:23,456 --> 00:01:26,789
            re.compile(r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})'),
            # Bracket format: [00:01:23]
            re.compile(r'\[(\d{1,2}:\d{2}:\d{2}(?:[.,]\d{3})?)\]'),
            # Parentheses: (00:01:23)
            re.compile(r'\((\d{1,2}:\d{2}:\d{2}(?:[.,]\d{3})?)\)'),
            # Simple format: 00:01:23
            re.compile(r'(?:^|\s)(\d{1,2}:\d{2}:\d{2}(?:[.,]\d{3})?)(?:\s|$)'),
        ]
        
        self.speaker_patterns = [
            re.compile(r'^([A-Z][A-Z\s]{1,25}):\s*', re.MULTILINE),
            re.compile(r'^([A-Z]{2,}(?:\s+[A-Z]{2,})*)\s*$', re.MULTILINE),
        ]

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 1 token = 0.75 words)"""
        return int(len(text.split()) * 1.33)

    def detect_encoding(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read(5000)
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'

    def read_file_content(self, file_path: str) -> str:
        encoding = self.detect_encoding(file_path)
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                content = file.read()
                logger.info(f"Read file: {Path(file_path).name} ({len(content):,} characters)")
                return content
        except Exception as e:
            for fallback in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=fallback, errors='ignore') as file:
                        return file.read()
                except:
                    continue
            raise Exception(f"Could not read file: {file_path}")

    def convert_timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert timestamp to seconds for ad placement calculations"""
        # Remove common separators and normalize
        timestamp = timestamp.replace(',', '.').strip()
        
        # Handle different formats
        if '-->' in timestamp:
            # Take start time from SRT format
            timestamp = timestamp.split('-->')[0].strip()
        
        # Parse HH:MM:SS.mmm or MM:SS or HH:MM:SS
        parts = timestamp.split(':')
        try:
            if len(parts) == 3:  # HH:MM:SS(.mmm)
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds_parts = parts[2].split('.')
                seconds = int(seconds_parts[0])
                milliseconds = int(seconds_parts[1][:3]) if len(seconds_parts) > 1 else 0
                return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
            elif len(parts) == 2:  # MM:SS
                minutes = int(parts[0])
                seconds = int(parts[1].split('.')[0])
                return minutes * 60 + seconds
        except (ValueError, IndexError):
            pass
        
        return 0.0

    def extract_timestamps_detailed(self, text: str) -> List[Dict]:
        """Extract ALL timestamps with detailed context for ad placement"""
        timestamps = []
        seen_positions = set()
        
        for pattern in self.timestamp_patterns:
            for match in pattern.finditer(text):
                if match.start() not in seen_positions:
                    timestamp_str = match.group(1) if len(match.groups()) == 1 else f"{match.group(1)} --> {match.group(2)}"
                    
                    # Get context around timestamp
                    context_start = max(0, match.start() - 200)
                    context_end = min(len(text), match.end() + 200)
                    context = text[context_start:context_end].strip()
                    
                    # Find scene/dialogue context
                    lines_before = text[:match.start()].split('\n')[-3:]
                    lines_after = text[match.end():].split('\n')[:3]
                    
                    timestamp_data = {
                        'timestamp': timestamp_str,
                        'position_in_text': match.start(),
                        'seconds': self.convert_timestamp_to_seconds(timestamp_str),
                        'context_before': ' '.join([line.strip() for line in lines_before if line.strip()])[-100:],
                        'context_after': ' '.join([line.strip() for line in lines_after if line.strip()])[:100],
                        'full_context': context,
                        'scene_break_potential': self._assess_scene_break(context)
                    }
                    
                    timestamps.append(timestamp_data)
                    seen_positions.add(match.start())
        
        return sorted(timestamps, key=lambda x: x['seconds'])

    def _assess_scene_break(self, context: str) -> float:
        """Assess if timestamp represents a good ad break point (0-1 score)"""
        score = 0.5  # Base score
        
        context_upper = context.upper()
        
        # Scene transition indicators
        scene_indicators = [
            'FADE OUT', 'FADE IN', 'CUT TO', 'INT.', 'EXT.',
            'LATER', 'MEANWHILE', 'END SCENE', '- DAY', '- NIGHT'
        ]
        
        for indicator in scene_indicators:
            if indicator in context_upper:
                score += 0.2
        
        # Dialogue endings (good for ads)
        if any(punct in context for punct in ['.', '!', '?']):
            score += 0.1
        
        # Action descriptions (less ideal for ads)
        action_words = ['RUNNING', 'FIGHTING', 'SCREAMING', 'CRYING']
        if any(word in context_upper for word in action_words):
            score -= 0.2
        
        return min(1.0, max(0.0, score))

    def extract_speakers(self, text: str) -> Tuple[List[str], Dict]:
        """Extract ALL speakers with comprehensive stats"""
        speakers = set()
        speaker_stats = {}
        speaker_dialogues = {}
        
        for pattern in self.speaker_patterns:
            for match in pattern.finditer(text):
                speaker = match.group(1).strip().upper()
                if 2 <= len(speaker) <= 25 and speaker.replace(' ', '').isalpha():
                    speakers.add(speaker)
                    if speaker not in speaker_stats:
                        speaker_stats[speaker] = {
                            'dialogue_count': 0, 
                            'total_words': 0,
                            'first_appearance_pos': match.start(),
                            'last_appearance_pos': match.start()
                        }
                        speaker_dialogues[speaker] = []
                    
                    speaker_stats[speaker]['dialogue_count'] += 1
                    speaker_stats[speaker]['last_appearance_pos'] = match.start()
                    
                    # Extract dialogue context
                    line_end = text.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(text)
                    dialogue = text[match.end():line_end].strip()
                    
                    if dialogue:
                        speaker_stats[speaker]['total_words'] += len(dialogue.split())
                        if len(speaker_dialogues[speaker]) < 3:  # Keep top 3 sample dialogues
                            speaker_dialogues[speaker].append(dialogue[:150])
        
        # Classify speakers and add additional stats
        for speaker, stats in speaker_stats.items():
            avg_words = stats['total_words'] / stats['dialogue_count'] if stats['dialogue_count'] > 0 else 0
            
            # Classify importance
            if stats['dialogue_count'] > 20:
                stats['importance'] = 'lead'
            elif stats['dialogue_count'] > 8:
                stats['importance'] = 'supporting'
            else:
                stats['importance'] = 'minor'
            
            stats['avg_words_per_dialogue'] = round(avg_words, 1)
            stats['sample_dialogues'] = speaker_dialogues.get(speaker, [])
            stats['screen_time_percentage'] = round(
                (stats['last_appearance_pos'] - stats['first_appearance_pos']) / len(text) * 100, 2
            )
        
        return sorted(list(speakers)), speaker_stats

    def clean_text(self, text: str, preserve_structure: bool = True) -> str:
        """Clean text with different levels based on usage"""
        cleaned = text
        
        # Remove HTML and problematic formatting
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        cleaned = re.sub(r'\{[^}]+\}', '', cleaned)
        
        if not preserve_structure:
            # More aggressive cleaning for NLP
            for pattern in self.timestamp_patterns:
                cleaned = pattern.sub('', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]*\n', '\n', cleaned)
        
        return cleaned.strip()

    def create_llm_chunks(self, text: str, timestamps: List[Dict], speakers: Dict,
                         model: str = 'claude-sonnet-4', overlap_ratio: float = 0.15) -> List[Dict]:
        """Create LLM-optimized chunks with full context"""
        
        max_tokens = self.model_limits.get(model, 100000)
        chunk_token_limit = int(max_tokens * 0.65)  # Use 65% for safety
        
        estimated_total_tokens = self.estimate_tokens(text)
        
        if estimated_total_tokens <= chunk_token_limit:
            # Single chunk - include ALL context
            return [{
                'chunk_id': 1,
                'text': text,
                'token_estimate': estimated_total_tokens,
                'timestamps_in_chunk': [ts['timestamp'] for ts in timestamps],
                'speakers_in_chunk': list(speakers.keys()),
                'ad_break_opportunities': [
                    {
                        'timestamp': ts['timestamp'],
                        'seconds': ts['seconds'],
                        'break_score': ts['scene_break_potential'],
                        'context': ts['context_after'][:100]
                    }
                    for ts in timestamps if ts['scene_break_potential'] > 0.7
                ],
                'is_complete_file': True,
                'total_chunks': 1
            }]
        
        # Multi-chunk processing
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        chunks = []
        current_chunk_paras = []
        current_tokens = 0
        chunk_id = 1
        
        overlap_paras = max(1, int(len(paragraphs) * overlap_ratio))
        
        for para in paragraphs:
            para_tokens = self.estimate_tokens(para)
            
            if current_tokens + para_tokens > chunk_token_limit and current_chunk_paras:
                # Create chunk
                chunk_text = "\n\n".join(current_chunk_paras)
                chunk_start = text.find(current_chunk_paras[0])
                chunk_end = text.find(current_chunk_paras[-1]) + len(current_chunk_paras[-1])
                
                # Find timestamps in this chunk
                chunk_timestamps = [
                    ts for ts in timestamps 
                    if chunk_start <= ts['position_in_text'] <= chunk_end
                ]
                
                # Find speakers in this chunk
                chunk_speakers = [
                    speaker for speaker, stats in speakers.items()
                    if any(sample in chunk_text.upper() for sample in [speaker])
                ]
                
                # Find ad opportunities
                ad_opportunities = [
                    {
                        'timestamp': ts['timestamp'],
                        'seconds': ts['seconds'],
                        'break_score': ts['scene_break_potential'],
                        'context': ts['context_after'][:100]
                    }
                    for ts in chunk_timestamps if ts['scene_break_potential'] > 0.6
                ]
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'token_estimate': current_tokens,
                    'timestamps_in_chunk': [ts['timestamp'] for ts in chunk_timestamps],
                    'speakers_in_chunk': chunk_speakers,
                    'ad_break_opportunities': ad_opportunities,
                    'is_complete_file': False,
                    'chunk_context': f"Part {chunk_id} of full transcript"
                })
                
                chunk_id += 1
                
                # Create overlap
                overlap_start = max(0, len(current_chunk_paras) - overlap_paras)
                current_chunk_paras = current_chunk_paras[overlap_start:] + [para]
                current_tokens = sum(self.estimate_tokens(p) for p in current_chunk_paras)
            else:
                current_chunk_paras.append(para)
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk_paras:
            chunk_text = "\n\n".join(current_chunk_paras)
            chunk_start = text.find(current_chunk_paras[0])
            chunk_end = text.find(current_chunk_paras[-1]) + len(current_chunk_paras[-1])
            
            chunk_timestamps = [
                ts for ts in timestamps 
                if chunk_start <= ts['position_in_text'] <= chunk_end
            ]
            
            chunk_speakers = [
                speaker for speaker, stats in speakers.items()
                if any(sample in chunk_text.upper() for sample in [speaker])
            ]
            
            ad_opportunities = [
                {
                    'timestamp': ts['timestamp'],
                    'seconds': ts['seconds'],
                    'break_score': ts['scene_break_potential'],
                    'context': ts['context_after'][:100]
                }
                for ts in chunk_timestamps if ts['scene_break_potential'] > 0.6
            ]
            
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'token_estimate': current_tokens,
                'timestamps_in_chunk': [ts['timestamp'] for ts in chunk_timestamps],
                'speakers_in_chunk': chunk_speakers,
                'ad_break_opportunities': ad_opportunities,
                'is_complete_file': False,
                'chunk_context': f"Part {chunk_id} of full transcript"
            })
        
        # Update total chunks count
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
        
        return chunks

    def create_nlp_data(self, text: str, speakers: Dict, timestamps: List[Dict]) -> Dict:
        """Create focused NLP processing data"""
        
        # Clean text for NLP
        clean_text = self.clean_text(text, preserve_structure=False)
        sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if s.strip() and len(s) > 10]
        
        # Extract key sentences (beginning, middle, end + dialogue-rich)
        key_sentences = []
        total_sentences = len(sentences)
        
        if total_sentences > 0:
            # Beginning and end
            key_sentences.extend(sentences[:5])
            if total_sentences > 15:
                mid_start = total_sentences // 2 - 3
                key_sentences.extend(sentences[mid_start:mid_start + 6])
            key_sentences.extend(sentences[-5:])
            
            # Dialogue-heavy sentences
            speaker_names = list(speakers.keys())
            dialogue_sentences = [
                s for s in sentences 
                if any(name in s.upper() for name in speaker_names[:5])  # Top 5 speakers only
            ][:10]  # Limit to 10
            
            key_sentences.extend(dialogue_sentences)
        
        # Remove duplicates
        seen = set()
        unique_sentences = []
        for sentence in key_sentences:
            if sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        
        # Create focused summary
        condensed_text = '. '.join(unique_sentences[:30])  # Limit to 30 sentences
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases_advanced(clean_text, speakers)
        
        # Timestamp analysis for NLP
        timestamp_analysis = {
            'total_timestamps': len(timestamps),
            'good_ad_breaks': len([ts for ts in timestamps if ts['scene_break_potential'] > 0.7]),
            'average_time_between_breaks': 0,
            'peak_ad_opportunities': []
        }
        
        if len(timestamps) > 1:
            time_intervals = []
            for i in range(1, len(timestamps)):
                interval = timestamps[i]['seconds'] - timestamps[i-1]['seconds']
                time_intervals.append(interval)
            timestamp_analysis['average_time_between_breaks'] = round(
                sum(time_intervals) / len(time_intervals), 2
            ) if time_intervals else 0
            
            # Find peak ad opportunities
            timestamp_analysis['peak_ad_opportunities'] = [
                {
                    'timestamp': ts['timestamp'],
                    'seconds': ts['seconds'],
                    'score': ts['scene_break_potential'],
                    'reason': 'Scene transition detected' if ts['scene_break_potential'] > 0.8 else 'Natural break'
                }
                for ts in sorted(timestamps, key=lambda x: x['scene_break_potential'], reverse=True)[:5]
            ]
        
        return {
            'condensed_text': condensed_text,
            'key_phrases': key_phrases,
            'speaker_summary': {
                name: {
                    'importance': stats['importance'],
                    'dialogue_count': stats['dialogue_count'],
                    'sample_dialogue': stats['sample_dialogues'][0] if stats['sample_dialogues'] else ""
                }
                for name, stats in sorted(speakers.items(), 
                                        key=lambda x: x[1]['dialogue_count'], 
                                        reverse=True)[:8]  # Top 8 speakers for NLP
            },
            'timestamp_analysis': timestamp_analysis,
            'content_metrics': {
                'total_words': len(clean_text.split()),
                'dialogue_density': len([s for s in sentences if ':' in s]) / len(sentences) if sentences else 0,
                'estimated_tokens': self.estimate_tokens(condensed_text),
                'processing_complexity': 'high' if len(speakers) > 10 else 'medium' if len(speakers) > 5 else 'low'
            }
        }

    def _extract_key_phrases_advanced(self, text: str, speakers: Dict) -> List[str]:
        """Advanced key phrase extraction"""
        # Proper nouns and capitalized words
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter out speaker names to get locations, objects, etc.
        speaker_names = set(speakers.keys())
        filtered_phrases = []
        
        word_freq = {}
        for phrase in proper_nouns:
            if phrase.upper() not in speaker_names and len(phrase) > 2:
                word_freq[phrase] = word_freq.get(phrase, 0) + 1
        
        # Return top phrases by frequency
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:15]

    def generate_file_hash(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def process_file(self, file_path: str, model: str = 'claude-sonnet-4') -> Tuple[Dict, Dict]:
        """Process file and return TWO separate JSONs: one for LLM, one for NLP"""
        start_time = datetime.now()
        
        if not os.path.isabs(file_path):
            file_path = str(self.project_root / file_path)
        
        logger.info(f"Processing: {Path(file_path).name}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read and extract data
        original_text = self.read_file_content(file_path)
        file_hash = self.generate_file_hash(original_text)
        
        timestamps = self.extract_timestamps_detailed(original_text)
        speakers, speaker_stats = self.extract_speakers(original_text)
        
        # Clean text for LLM (preserve structure)
        cleaned_text_llm = self.clean_text(original_text, preserve_structure=True)
        
        # Create LLM chunks
        llm_chunks = self.create_llm_chunks(cleaned_text_llm, timestamps, speaker_stats, model)
        
        # Create NLP data
        nlp_data = self.create_nlp_data(original_text, speaker_stats, timestamps)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Common file info
        file_info = {
            'name': os.path.basename(file_path),
            'hash': file_hash,
            'size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
            'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'processing_time_seconds': round(processing_time, 2)
        }
        
        # LLM JSON - Complete context for analysis
        llm_json = {
            'file_info': file_info.copy(),
            'model_optimized_for': model,
            'processing_type': 'LLM_ANALYSIS',
            'content_stats': {
                'total_words': len(cleaned_text_llm.split()),
                'estimated_total_tokens': sum(chunk['token_estimate'] for chunk in llm_chunks),
                'total_speakers': len(speakers),
                'total_timestamps': len(timestamps),
                'estimated_duration_minutes': round(len(cleaned_text_llm.split()) / 165, 2)
            },
            'chunks_for_processing': llm_chunks,
            'all_timestamps_with_ad_context': [
                {
                    'timestamp': ts['timestamp'],
                    'seconds': ts['seconds'],
                    'ad_break_score': ts['scene_break_potential'],
                    'context_before': ts['context_before'],
                    'context_after': ts['context_after']
                }
                for ts in timestamps
            ],
            'speaker_context': speaker_stats,
            'processing_instructions': {
                'chunk_strategy': 'overlapping' if len(llm_chunks) > 1 else 'single',
                'total_chunks': len(llm_chunks),
                'model_token_limit': self.model_limits.get(model, 'unknown'),
                'preserve_context': True
            }
        }
        
        # NLP JSON - Focused data for cross-validation
        nlp_json = {
            'file_info': file_info.copy(),
            'processing_type': 'NLP_VALIDATION',
            'focused_content': nlp_data,
            'validation_targets': {
                'key_speakers_to_verify': list(nlp_data['speaker_summary'].keys()),
                'timestamp_accuracy_check': len(timestamps) > 0,
                'sentiment_analysis_ready': True,
                'keyword_extraction_ready': True
            },
            'cross_validation_metrics': {
                'content_complexity': nlp_data['content_metrics']['processing_complexity'],
                'expected_processing_time': 'fast',
                'confidence_factors': {
                    'speaker_detection': 'high' if len(speakers) > 3 else 'medium',
                    'timestamp_detection': 'high' if len(timestamps) > 5 else 'medium',
                    'content_structure': 'high' if nlp_data['content_metrics']['dialogue_density'] > 0.3 else 'medium'
                }
            }
        }
        
        # Save both JSONs
        base_name = f"{file_hash}_{Path(file_path).stem}"
        
        llm_file = self.processed_dir / f"{base_name}_LLM.json"
        nlp_file = self.processed_dir / f"{base_name}_NLP.json"
        
        with open(llm_file, 'w', encoding='utf-8') as f:
            json.dump(llm_json, f, indent=2, ensure_ascii=False)
            
        with open(nlp_file, 'w', encoding='utf-8') as f:
            json.dump(nlp_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved LLM JSON: {llm_file.name}")
        logger.info(f"✅ Saved NLP JSON: {nlp_file.name}")
        logger.info(f"📊 Timestamps found: {len(timestamps)}, Speakers: {len(speakers)}, Chunks: {len(llm_chunks)}")
        
        return llm_json, nlp_json

    def process_all_uploads(self, model: str = 'claude-sonnet-4') -> List[Tuple[Dict, Dict]]:
        """Process all files and return pairs of (LLM_JSON, NLP_JSON)"""
        results = []
        upload_files = []
        
        for file_path in self.uploads_dir.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                upload_files.append(str(file_path))
        
        if not upload_files:
            logger.warning("No files found in uploads directory")
            return results
        
        logger.info(f"Found {len(upload_files)} files to process for {model}")
        
        for file_path in upload_files:
            try:
                llm_json, nlp_json = self.process_file(file_path, model)
                results.append((llm_json, nlp_json))
                print(f"✅ {Path(file_path).name} → LLM: {len(llm_json['chunks_for_processing'])} chunks, NLP: Ready")
            except Exception as e:
                logger.error(f"Failed to process {Path(file_path).name}: {e}")
                print(f"❌ Failed: {Path(file_path).name} - {e}")
        
        return results

def test_dual_json_processing():
    """Test the dual JSON processing approach"""
    print("🚀 Testing Dual JSON Processing (LLM + NLP)")
    print("=" * 60)
    
    handler = FileHandler()
    
    # Create sample with timestamps for testing
    upload_files = list(handler.uploads_dir.glob("*"))
    upload_files = [f for f in upload_files if f.suffix.lower() in handler.supported_formats]
    
    if not upload_files:
        sample_with_timestamps = '''[00:00:12] INT. COFFEE SHOP - DAY

SARAH enters, looking around nervously.

SARAH
Where is he? I'm already five minutes late.

[00:00:45] JOHN waves from corner table.

JOHN
Sarah! Over here. I got your usual - vanilla latte with an extra shot.

SARAH
Thanks, John. Traffic was insane on Broadway.

[00:01:30] They sit across from each other. Awkward silence.

JOHN
So... about last night's dinner conversation.

SARAH
John, can we not do this here? People might overhear us.

[00:02:15] JOHN
We need to talk about the promotion offer. I know I reacted badly.

SARAH
You didn't just react badly - you completely dismissed my career goals.

JOHN
I was scared. Scared of losing you to some fancy job in Chicago.

[00:03:00] SARAH
It's not "some job" - this is my dream position, John.

SARAH
(standing up)
I think we need a break to figure out what we both really want.

[00:03:30] SARAH exits. JOHN sits alone with two untouched lattes.

FADE OUT.'''
        
        sample_path = handler.uploads_dir / "timestamped_script.txt"
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample_with_timestamps)
        print(f"📝 Created timestamped sample: {sample_path.name}")
    
    # Process all files
    results = handler.process_all_uploads('claude-sonnet-4')
    
    print(f"\n📊 PROCESSING RESULTS:")
    print("-" * 50)
    
    for llm_json, nlp_json in results:
        print(f"\n🎬 File: {llm_json['file_info']['name']}")
        print(f"   📄 Size: {llm_json['file_info']['size_mb']} MB")
        print(f"   ⏱️  Processing: {llm_json['file_info']['processing_time_seconds']}s")
        
        # LLM Data
        print(f"\n   🤖 LLM DATA:")
        print(f"      Chunks: {len(llm_json['chunks_for_processing'])}")
        print(f"      Total Tokens: {llm_json['content_stats']['estimated_total_tokens']:,}")
        print(f"      Timestamps: {llm_json['content_stats']['total_timestamps']}")
        print(f"      Speakers: {llm_json['content_stats']['total_speakers']}")
        
        # Show timestamp details
        if llm_json['all_timestamps_with_ad_context']:
            print(f"      🎯 Ad Break Opportunities:")
            good_breaks = [ts for ts in llm_json['all_timestamps_with_ad_context'] if ts['ad_break_score'] > 0.7]
            for i, ts in enumerate(good_breaks[:3], 1):
                print(f"         {i}. {ts['timestamp']} (score: {ts['ad_break_score']:.2f})")
        
        # NLP Data
        print(f"\n   🔬 NLP DATA:")
        print(f"      Condensed Tokens: {nlp_json['focused_content']['content_metrics']['estimated_tokens']}")
        print(f"      Key Speakers: {len(nlp_json['focused_content']['speaker_summary'])}")
        print(f"      Key Phrases: {len(nlp_json['focused_content']['key_phrases'])}")
        print(f"      Complexity: {nlp_json['focused_content']['content_metrics']['processing_complexity']}")
        
        # Show key phrases
        if nlp_json['focused_content']['key_phrases']:
            phrases = nlp_json['focused_content']['key_phrases'][:5]
            print(f"      Top Phrases: {', '.join(phrases)}")
        
        # Show confidence factors
        confidence = nlp_json['cross_validation_metrics']['confidence_factors']
        print(f"      Confidence: Speaker-{confidence['speaker_detection']}, Timestamp-{confidence['timestamp_detection']}")
    
    # Show saved files
    processed_files = list(handler.processed_dir.glob("*.json"))
    llm_files = [f for f in processed_files if "_LLM.json" in f.name]
    nlp_files = [f for f in processed_files if "_NLP.json" in f.name]
    
    print(f"\n💾 FILES SAVED:")
    print(f"   🤖 LLM JSONs: {len(llm_files)}")
    for f in llm_files:
        size_kb = f.stat().st_size / 1024
        print(f"      📄 {f.name} ({size_kb:.1f} KB)")
    
    print(f"   🔬 NLP JSONs: {len(nlp_files)}")
    for f in nlp_files:
        size_kb = f.stat().st_size / 1024
        print(f"      📄 {f.name} ({size_kb:.1f} KB)")
    
    print(f"\n✅ Summary: {len(results)} files processed → {len(llm_files)} LLM + {len(nlp_files)} NLP JSONs")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process specific file
        file_path = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else 'claude-sonnet-4'
        
        handler = FileHandler()
        try:
            llm_json, nlp_json = handler.process_file(file_path, model)
            print("✅ LLM JSON created with full context")
            print("✅ NLP JSON created with focused data")
            print(f"📊 Timestamps: {len(llm_json['all_timestamps_with_ad_context'])}")
            print(f"📊 Chunks: {len(llm_json['chunks_for_processing'])}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        # Test with sample data
        test_dual_json_processing()