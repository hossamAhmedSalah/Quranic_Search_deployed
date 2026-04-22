"""
Build full Quranic embeddings in ChromaDB.

This script creates full-Quran embeddings for semantic search using silma-ai model.
It generates two collections:
  - quran_verses: embeddings of verse text
  - quran_tafsir:  embeddings of tafsir (interpretation) text

These are then used by app_plus.py without needing to re-embed at runtime.
"""

import pandas as pd
import numpy as np
import chromadb
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
BASE_PATH = Path('Data')
MODEL_NAME = 'silma-ai/silma-embeddding-sts-0.1'
CHROMADB_DIR = BASE_PATH / '.chromadb'
BATCH_SIZE = 100

SURAH_NAMES = {
    1:"الفاتحة",2:"البقرة",3:"آل عمران",4:"النساء",5:"المائدة",
    6:"الأنعام",7:"الأعراف",8:"الأنفال",9:"التوبة",10:"يونس",
    11:"هود",12:"يوسف",13:"الرعد",14:"إبراهيم",15:"الحجر",
    16:"النحل",17:"الإسراء",18:"الكهف",19:"مريم",20:"طه",
    21:"الأنبياء",22:"الحج",23:"المؤمنون",24:"النور",25:"الفرقان",
    26:"الشعراء",27:"النمل",28:"القصص",29:"العنكبوت",30:"الروم",
    31:"لقمان",32:"السجدة",33:"الأحزاب",34:"سبأ",35:"فاطر",
    36:"يس",37:"الصافات",38:"ص",39:"الزمر",40:"غافر",
    41:"فصلت",42:"الشورى",43:"الزخرف",44:"الدخان",45:"الجاثية",
    46:"الأحقاف",47:"محمد",48:"الفتح",49:"الحجرات",50:"ق",
    51:"الذاريات",52:"الطور",53:"النجم",54:"القمر",55:"الرحمن",
    56:"الواقعة",57:"الحديد",58:"المجادلة",59:"الحشر",60:"الممتحنة",
    61:"الصف",62:"الجمعة",63:"المنافقون",64:"التغابن",65:"الطلاق",
    66:"التحريم",67:"الملك",68:"القلم",69:"الحاقة",70:"المعارج",
    71:"نوح",72:"الجن",73:"المزمل",74:"المدثر",75:"القيامة",
    76:"الإنسان",77:"المرسلات",78:"النبأ",79:"النازعات",80:"عبس",
    81:"التكوير",82:"الانفطار",83:"المطففين",84:"الانشقاق",85:"البروج",
    86:"الطارق",87:"الأعلى",88:"الغاشية",89:"الفجر",90:"البلد",
    91:"الشمس",92:"الليل",93:"الضحى",94:"الشرح",95:"التين",
    96:"العلق",97:"القدر",98:"البينة",99:"الزلزلة",100:"العاديات",
    101:"القارعة",102:"التكاثر",103:"العصر",104:"الهمزة",105:"الفيل",
    106:"قريش",107:"الماعون",108:"الكوثر",109:"الكافرون",110:"النصر",
    111:"المسد",112:"الإخلاص",113:"الفلق",114:"الناس",
}

# ─────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────
print("📦 Loading Quranic data...")
df_linguistic = pd.read_csv(BASE_PATH / 'Linguistic' / 'Quranic.csv', sep='\t', encoding='utf-16')
df_interpretation = pd.read_csv(BASE_PATH / 'Interpretation' / 'ar.muyassar.csv', encoding='utf-8')

print(f"   ✓ Linguistic data: {len(df_linguistic)} rows")
print(f"   ✓ Interpretation data: {len(df_interpretation)} rows")

# ─────────────────────────────────────────────────────────────────
# BUILD VERSE-LEVEL DATA
# ─────────────────────────────────────────────────────────────────
print("\n🔨 Building verse-level data...")

# Remove implicit pronouns (word_id = 0)
df_ling = df_linguistic[df_linguistic['word_id'] > 0].copy()

# Verse text: join words
verse_text = (
    df_ling.groupby(['chapter_id', 'verse_id'])['uthmani_token']
    .apply(lambda x: ' '.join(x.astype(str)))
    .reset_index()
    .rename(columns={'uthmani_token': 'verse_text'})
)

# Verse translation
def join_trans(x):
    parts = x.dropna().astype(str)
    parts = parts[(parts.str.strip() != '_') & (parts.str.strip() != '')]
    return ' '.join(parts.unique())

verse_trans = (
    df_ling.groupby(['chapter_id', 'verse_id'])['trans']
    .apply(join_trans)
    .reset_index()
    .rename(columns={'trans': 'verse_trans'})
)

verses = verse_text.merge(verse_trans, on=['chapter_id', 'verse_id'])
verses['surah_name'] = verses['chapter_id'].map(SURAH_NAMES)
verses['ref'] = verses.apply(
    lambda r: f"{r['surah_name']} ({r['chapter_id']}:{r['verse_id']})", axis=1
)

print(f"   ✓ Total verses: {len(verses)}")

# ─────────────────────────────────────────────────────────────────
# ADD INTERPRETATION DATA
# ─────────────────────────────────────────────────────────────────
# Map interpretation by surah + verse
df_interpretation['Surah'] = pd.to_numeric(
    df_interpretation['Surah'], errors='coerce'
).fillna(0).astype(int)
df_interpretation['Verse'] = pd.to_numeric(
    df_interpretation['Verse'], errors='coerce'
).fillna(0).astype(int)

interp_map = {}
for _, row in df_interpretation.iterrows():
    key = (int(row['Surah']), int(row['Verse']))
    interp_map[key] = str(row.get('Tafsir', ''))[:500]  # Truncate to 500 chars

verses['tafsir'] = verses.apply(
    lambda r: interp_map.get((r['chapter_id'], r['verse_id']), ''),
    axis=1
)

print(f"   ✓ Verses with tafsir: {(verses['tafsir'].str.len() > 0).sum()}")

# ─────────────────────────────────────────────────────────────────
# LOAD MODEL & CREATE CHROMADB CLIENT
# ─────────────────────────────────────────────────────────────────
print(f"\n🤖 Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print(f"   ✓ Embedding dimension: {model.get_sentence_embedding_dimension()}")

CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n📂 Initializing ChromaDB at: {CHROMADB_DIR.resolve()}")
client = chromadb.PersistentClient(path=str(CHROMADB_DIR))

# Delete existing collections if they exist
existing = [c.name for c in client.list_collections()]
for col_name in ['quran_verses', 'quran_tafsir']:
    if col_name in existing:
        print(f"   ⚠️  Deleting existing collection: {col_name}")
        client.delete_collection(col_name)

# ─────────────────────────────────────────────────────────────────
# CREATE VERSE COLLECTION
# ─────────────────────────────────────────────────────────────────
print(f"\n✍️  Creating 'quran_verses' collection...")
verse_col = client.create_collection(
    name="quran_verses",
    metadata={"hnsw:space": "cosine"}
)

# Encode and add verses in batches
batch_size = BATCH_SIZE
total_verses = len(verses)

for i in tqdm(range(0, total_verses, batch_size), desc="Encoding verses", unit="batch"):
    batch = verses.iloc[i:i+batch_size]
    
    # Encode verse texts
    verse_texts = batch['verse_text'].tolist()
    embeddings = model.encode(verse_texts, normalize_embeddings=True).tolist()
    
    # Prepare metadata
    metadatas = []
    documents = []
    ids = []
    
    for idx, (_, row) in enumerate(batch.iterrows()):
        doc_id = f"v_{row['chapter_id']}_{row['verse_id']}"
        ids.append(doc_id)
        documents.append(row['verse_text'])
        metadatas.append({
            'chapter_id': int(row['chapter_id']),
            'verse_id': int(row['verse_id']),
            'ref': row['ref'],
            'verse_text': row['verse_text'],
            'verse_trans': row['verse_trans'],
            'tafsir': row['tafsir'],
            'root_bag': '',  # Optional: could add root bag here
        })
    
    verse_col.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

print(f"   ✅ 'quran_verses' collection created with {total_verses} verses!")

# ─────────────────────────────────────────────────────────────────
# CREATE TAFSIR COLLECTION
# ─────────────────────────────────────────────────────────────────
print(f"\n✍️  Creating 'quran_tafsir' collection...")
tafsir_col = client.create_collection(
    name="quran_tafsir",
    metadata={"hnsw:space": "cosine"}
)

# Filter verses with tafsir
verses_with_tafsir = verses[verses['tafsir'].str.len() > 0].copy()
total_tafsir = len(verses_with_tafsir)
print(f"   Processing {total_tafsir} verses with tafsir...")

for i in tqdm(range(0, total_tafsir, batch_size), desc="Encoding tafsir", unit="batch"):
    batch = verses_with_tafsir.iloc[i:i+batch_size]
    
    # Encode tafsir texts
    tafsir_texts = batch['tafsir'].tolist()
    embeddings = model.encode(tafsir_texts, normalize_embeddings=True).tolist()
    
    # Prepare metadata
    metadatas = []
    documents = []
    ids = []
    
    for idx, (_, row) in enumerate(batch.iterrows()):
        doc_id = f"t_{row['chapter_id']}_{row['verse_id']}"
        ids.append(doc_id)
        documents.append(row['tafsir'])
        metadatas.append({
            'chapter_id': int(row['chapter_id']),
            'verse_id': int(row['verse_id']),
            'ref': row['ref'],
            'verse_text': row['verse_text'],
            'verse_trans': row['verse_trans'],
            'tafsir': row['tafsir'],
            'root_bag': '',
        })
    
    tafsir_col.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

print(f"   ✅ 'quran_tafsir' collection created with {total_tafsir} verses!")

# ─────────────────────────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────────────────────────
print(f"\n✨ Verification:")
all_cols = client.list_collections()
for col in all_cols:
    if col.name in ['quran_verses', 'quran_tafsir']:
        print(f"   ✅ {col.name}: {col.count()} documents")

print(f"\n🎉 ✅ Full Quranic embeddings successfully saved to: {CHROMADB_DIR.resolve()}")
print(f"   App will now use these pre-saved embeddings without re-embedding!")
