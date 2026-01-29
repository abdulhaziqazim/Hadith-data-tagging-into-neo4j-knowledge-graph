import re
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import ollama
import ast  # For safely parsing string list

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Clean isnad by removing start keywords and parentheses
def clean_isnad(isnad):
    start_keywords = [
        "Narrated by", "Narrated", "It is reported by", "Reported by", "It ", "The", "My", "If", "He", "She",
        "On the authority of", "He said", "She said", "Narrates", "Narration of", "The", "Related",
        "It was narrated from", "It was narrated that", "It was narrated by"
    ]
    for keyword in start_keywords:
        isnad = re.sub(rf"\b{re.escape(keyword)}\b", "", isnad, flags=re.IGNORECASE)

    # Remove parentheses and content inside
    isnad = re.sub(r"\([^)]*\)", "", isnad)
    return isnad.strip()

# Clean Hadith content of unnecessary characters
def clean_content(text):
    text = re.sub(r"\(\?\)", "", text)
    text = text.replace("?", "")
    return text.strip()

# Split isnad and content
def split_isnad_and_content(text):
    match = re.search(r"(It was narrated (from|by|that)[^:]*):", text, re.IGNORECASE)
    if match:
        isnad = match.group(1)
        content = text[match.end():].strip()
    else:
        split_keywords = ["say", "used", "told", " saying", " narrated", " reported", ":"]
        split_index = len(text)
        for keyword in split_keywords:
            idx = text.lower().find(keyword)
            if idx != -1:
                split_index = min(split_index, idx)
        isnad = text[:split_index].strip()
        content = text[split_index:].strip()
    return isnad, clean_content(content)

# Extract Narrators using spaCy NER and improved rule-based logic
def extract_narrators_from_isnad(isnad):
    isnad = clean_isnad(isnad)
    if ":" in isnad:
        isnad = isnad.split(":")[0].strip()

    parts = [p.strip() for p in re.split(r",| and ", isnad) if p.strip()]
    narrators = []
    name_prefixes = {"ibn", "bin", "ibnu", "bint", "ul", "al-", "as-", "b."}

    stopwords = set(STOP_WORDS) - {"said"}

    for part in parts:
        tokens = part.split()
        current_name = []
        i = 0
        while i < len(tokens):
            token = tokens[i].strip(".,:;!?\"“”‘’[]{}()")
            lower = token.lower()
            if lower in name_prefixes or re.match(r"^[A-Z'`‘].*", token):
                current_name.append(token)
                j = i + 1
                while j < len(tokens):
                    next_token = tokens[j].strip(".,:;!?\"“”‘’[]{}()")
                    if next_token.lower() in name_prefixes or re.match(r"^[A-Z'`‘].*", next_token):
                        current_name.append(next_token)
                        j += 1
                    else:
                        break
                full_name = " ".join(current_name).strip()
                # Discard if any token is a stopword or if the name itself is a stopword
                if not any(w.lower() in stopwords for w in current_name) and full_name.lower() not in stopwords:
                    narrators.append(full_name)
                current_name = []
                i = j
            else:
                i += 1

    # Normalize and deduplicate
    seen = set()
    final_narrators = []
    for name in narrators:
        name_cleaned = re.sub(r"\s+", " ", name.strip())
        name_cleaned = re.sub(r"^['`‘]", "", name_cleaned)
        if name_cleaned and name_cleaned.lower() not in stopwords and name_cleaned not in seen:
            seen.add(name_cleaned)
            final_narrators.append(name_cleaned)

    return final_narrators

# Use LLM to classify top 3 themes
def extract_theme(content, book_title):
    prompt = f"""You are a classifier that reads hadith content and suggests 3 most suitable themes. Example can be taken from this list:
Prayer, Fasting, Hajj, Marriage, Divorce, Zakat, Charity, Cleanliness, Family, Travel, Business, Foods, Inheritance, Manner, Sahabat, General.

Return ONLY a Python-style list 3 themes. No explanation.

Hadith content:
{content}

Book Title: {book_title}
"""
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

# Main tagging function
def tag_hadith(text, book_title):
    isnad, content = split_isnad_and_content(text)
    narrators = extract_narrators_from_isnad(isnad)
    themes_str = extract_theme(content, book_title)

    # Format narrators with double quotes
    narrators_formatted = "[" + ", ".join(f'"{n}"' for n in narrators) + "]"

    # Parse themes string safely to list and format with double quotes
    try:
        themes_list = ast.literal_eval(themes_str)
        if isinstance(themes_list, list):
            themes_formatted = "[" + ", ".join(f'"{t}"' for t in themes_list) + "]"
        else:
            themes_formatted = themes_str  # fallback if parsing fails
    except Exception:
        themes_formatted = themes_str  # fallback if parsing fails

    return {
        "Narrators": narrators_formatted,
        "Themes": themes_formatted
    }

# Apply on CSV
def process_csv(input_path, output_path, hadith_column='Hadith', book_column='Book Title'):
    df = pd.read_csv(input_path)

    narrators_list = []
    themes_list = []
    hadith_ids = []
    cleaned_hadiths = []

    for idx, row in df.iterrows():
        hadith = row[hadith_column]
        book = row[book_column]

        cleaned_hadith = clean_content(hadith)
        cleaned_hadiths.append(cleaned_hadith)

        result = tag_hadith(cleaned_hadith, book)

        narrators_list.append(result["Narrators"])
        themes_list.append(result["Themes"])
        hadith_ids.append(f"HAD{idx+1:04d}")

        print(f" Row {idx+1}/{len(df)} processed - Themes: {result['Themes']}")

    df["Hadith"] = cleaned_hadiths
    df["Hadith ID"] = hadith_ids
    df["Narrators"] = narrators_list
    df["Themes"] = themes_list

    ordered_columns = ["Hadith ID", "Book Title", "Collection", "Hadith", "Narrators", "Themes"]
    df = df[ordered_columns]

    df.to_csv(output_path, index=False)
    print(f"\nOutput saved to: {output_path}")

# Run
if __name__ == "__main__":
    input_csv = "test_sample_combined_hadith.csv"
    output_csv = "tagged_hadith_NEW6.csv"
    process_csv(input_csv, output_csv)
