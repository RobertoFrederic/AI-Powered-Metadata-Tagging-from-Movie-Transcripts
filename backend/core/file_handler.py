import os
import json
from pathlib import Path
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load spaCy English model
nlp = spacy.load("en_core_web_md")

# --- Screenplay-specific stopwords / scene indicators ---
FILM_STOPWORDS = {"int", "ext", "cut", "fade", "night", "day", "scene"}

# --- Helper Functions ---
def remove_scene_headers(text: str) -> str:
    return re.sub(r'^(INT|EXT)\..*?(\.|$)', '', text, flags=re.IGNORECASE | re.MULTILINE)

def clean_text(text: str) -> str:
    text = remove_scene_headers(text)
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.pos_ in ["NOUN", "PROPN", "ADJ"]
        and token.text.lower() not in STOP_WORDS
        and token.text.lower() not in FILM_STOPWORDS
        and len(token.text) > 2
    ]
    return " ".join(tokens)

def extract_characters(scene_text):
    doc = nlp(scene_text)
    characters = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
    return sorted(characters)

def consolidate_characters(char_list):
    """Combine variants of same character (simple heuristic)"""
    mapping = {}
    normalized = []
    for c in char_list:
        key = c.lower().replace('"', '').split()[0]  # first word lowercase
        if key not in mapping:
            mapping[key] = c
            normalized.append(c)
    return normalized

def extract_keywords(text, top_k=10, exclude_chars=None):
    text = remove_scene_headers(text)
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.pos_ in ["NOUN", "PROPN", "ADJ"]
        and token.text.lower() not in STOP_WORDS
        and token.text.lower() not in FILM_STOPWORDS
        and len(token.text) > 2
    ]
    if exclude_chars:
        tokens = [t for t in tokens if t.lower() not in [c.lower() for c in exclude_chars]]
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_keywords[:top_k]]

def extract_entities(scene_text):
    doc = nlp(scene_text)
    people = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
    places = {ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]}
    objects = {token.text for token in doc if token.pos_ == "NOUN" and token.text not in people and token.text not in places}
    return {"people": sorted(people), "places": sorted(places), "objects": sorted(objects)}

def detect_sensitive_content(scene_text):
    violence_words = {"kill", "gun", "fight", "shoot"}
    return {
        "violence": any(word.lower() in violence_words for word in scene_text.split()),
        "drugs": False,
        "profanity": False,
        "weapons": "gun" in scene_text.lower(),
        "sexual_content": False,
    }

# --- Processing ---
def process_for_llm(file_path, movie_id, title, year=None):
    """Combine full script into one large chunk for LLM"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().replace("\x00", " ").strip()  # remove null chars

    clean = clean_text(text)
    output = {
        "movie_id": movie_id,
        "title": title,
        "year": year or "Unknown",
        "script_text": text,
        "clean_text": clean
    }
    return output

def process_for_nlp(file_path, movie_id, title, year=None):
    """Refined NLP chunks with consolidated characters, keywords"""
    with open(file_path, "r", encoding="utf-8") as f:
        scenes = f.read().split("\n\n")  # naive split

    chunks = []
    all_characters = set()
    for idx, scene in enumerate(scenes, start=1):
        if not scene.strip():
            continue
        characters = consolidate_characters(extract_characters(scene))
        keywords = extract_keywords(scene, exclude_chars=characters)
        entities = extract_entities(scene)
        chunk = {
            "chunk_id": idx,
            "timestamp": {"start": None, "end": None},
            "text": scene.strip(),
            "clean_text": clean_text(scene),
            "keywords": keywords,
            "characters": characters,
            "locations": entities["places"],
            "entities": entities,
            "synopsis": scene.strip()[:200],  # simple snippet
            "sentiment": "neutral",
            "ad_placement": {
                "suitability": False,
                "reason": "Scene too short for ad break.",
                "recommended_timestamp": f"chunk_{idx}_end"
            },
            "embedding": [0.0]*10
        }
        chunks.append(chunk)
        all_characters.update(characters)

    output = {
        "movie_id": movie_id,
        "title": title,
        "year": year or "Unknown",
        "metadata": {"genre": [], "language": "en", "duration_minutes": None},
        "transcript": chunks,
        "stats": {"num_chunks": len(chunks), "num_characters": len(all_characters)}
    }
    return output

def main():
    uploads_dir = Path("data/uploads")
    llm_dir = Path("data/processed/LLM_jsons")
    nlp_dir = Path("data/processed/NLP_jsons")
    llm_dir.mkdir(parents=True, exist_ok=True)
    nlp_dir.mkdir(parents=True, exist_ok=True)

    for file in uploads_dir.glob("*.txt"):
        movie_id = file.stem
        title = file.stem.replace("_", " ").title()

        # LLM processing
        llm_output = process_for_llm(file, movie_id, title)
        with open(llm_dir / f"{file.stem}.json", "w", encoding="utf-8") as f:
            json.dump(llm_output, f, indent=2)

        # NLP processing
        nlp_output = process_for_nlp(file, movie_id, title)
        with open(nlp_dir / f"{file.stem}.json", "w", encoding="utf-8") as f:
            json.dump(nlp_output, f, indent=2)

        print(f"Processed → {file.name}: LLM & NLP JSONs saved.")

if __name__ == "__main__":
    main()
