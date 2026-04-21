"""
Quranic Search System — Streamlit Elite Edition
Root · Word · Lemma · Semantic search with Uthmanic Mushaf rendering

Standalone application that loads data and runs a comprehensive Quranic search GUI.
Semantic mode uses pre-saved ChromaDB embeddings (silma-ai/silma-embeddding-sts-0.1)
so no re-encoding is needed at runtime.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import re
import math
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  DATA PATHS
# ─────────────────────────────────────────────
BASE_PATH = Path('Data')

# Path to the persisted ChromaDB directory created from the notebook pipeline.
# Expected collections inside:
#   "quran_verses"  — verse text embeddings
#   "quran_tafsir"  — tafsir embeddings
# Each document must have metadata keys: chapter_id, verse_id, ref,
#   verse_text, verse_trans, tafsir, root_bag
# Try multiple possible paths in order of preference
_possible_chroma_paths = [
    Path('Data/.chromadb'),  # Inside Data folder
    Path('.chromadb'),       # Current directory
    Path('chroma_db'),       # Alternative name
]

CHROMA_PATH = None
for path in _possible_chroma_paths:
    if path.exists():
        CHROMA_PATH = path
        break

# Fallback to default if none exist
if CHROMA_PATH is None:
    CHROMA_PATH = _possible_chroma_paths[0]

CHROMA_VERSE_COLLECTION  = "quran_verses"
CHROMA_TAFSIR_COLLECTION = "quran_tafsir"
SEMANTIC_MODEL_NAME = "silma-ai/silma-embeddding-sts-0.1"

# ─────────────────────────────────────────────
#  PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="البحث القرآني | Quranic Search",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CONSTANTS & UTILITIES
# ─────────────────────────────────────────────
PAGE_SIZE = 20   # results per page

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

_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

def to_ar(n: int) -> str:
    """Convert integer to Arabic-Indic digit string."""
    return str(n).translate(_ARABIC_INDIC)


# ─────────────────────────────────────────────
#  ARABIC NORMALIZATION
# ─────────────────────────────────────────────
def normalize_arabic(text: str) -> str:
    """
    Comprehensive Arabic normalization for fuzzy matching.
    Removes diacritics, tatweel, and unifies character variants.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove tashkeel (harakat)
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]", "", text)
    
    # Remove tatweel (kashida)
    text = re.sub(r"\u0640", "", text)
    
    # Remove zero-width characters
    text = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]", "", text)
    
    # Unify character variants
    text = re.sub(r"[إأآٱا]", "ا", text)  # All alef variants → ا
    text = re.sub(r"ة", "ه", text)        # Taa marbuta → ha
    text = re.sub(r"ى", "ي", text)        # Alef maqsura → ya
    text = re.sub(r"ؤ", "و", text)        # Waw hamza → waw
    text = re.sub(r"ئ", "ي", text)        # Ya hamza → ya
    
    return text.strip()


# ─────────────────────────────────────────────
#  MORPHOLOGICAL TOOLS (PySarf)
# ─────────────────────────────────────────────
@st.cache_resource
def load_sarf():
    """Load PySarf analyzer (cached)."""
    from pysarf import PySarf
    return PySarf()

sarf = load_sarf()

@lru_cache(maxsize=20000)
def get_root_pysarf(word: str) -> str:
    """Extract root using PySarf."""
    if not isinstance(word, str) or not word.strip():
        return ""
    
    result = sarf.analyze(word)
    if result and result.root:
        return normalize_arabic(result.root)
    
    return normalize_arabic(word)


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and process Quranic data."""
    # Load linguistic data
    df_linguistic = pd.read_csv(
        BASE_PATH / 'Linguistic' / 'Quranic.csv', 
        sep='\t', encoding='utf-16'
    )
    
    # Load subjective and interpretation data (optional)
    df_subjective = pd.read_csv(
        BASE_PATH / 'Subjective' / 'Quranic_Tags.csv', 
        encoding='utf-8'
    )
    df_interpretation = pd.read_csv(
        BASE_PATH / 'Interpretation' / 'ar.muyassar.csv', 
        encoding='utf-8'
    )
    
    return df_linguistic, df_subjective, df_interpretation


@st.cache_data
def process_word_level(_df_linguistic):
    """Process linguistic data to word level."""
    # Remove implicit pronouns (word_id = 0)
    df_ling = _df_linguistic[_df_linguistic['word_id'] > 0].copy()
    df_ling['root_ar'] = df_ling['root_ar'].apply(
        lambda x: normalize_arabic(x) if pd.notna(x) else ""
    )
    df_ling['lemma_ar'] = df_ling['lemma_ar'].apply(
        lambda x: normalize_arabic(x) if pd.notna(x) else ""
    )
    
    # Aggregate by (chapter_id, verse_id, word_id)
    def get_word_root(series):
        vals = []
        for r in series:
            if pd.isna(r):
                continue
            r = str(r).strip()
            if not r or r == "-":
                continue
            vals.append(r)
        
        for r in vals:
            if r != "ـ":
                return r
        
        return "ـ" if "ـ" in vals else None
    
    word_level = (
        df_ling
        .groupby(["chapter_id", "verse_id", "word_id"], sort=True)
        .agg({
            "uthmani_token": lambda x: "".join(x.astype(str)),
            "root_ar": get_word_root,
            "lemma_ar": "first",
            "trans": "first",
            "pos": "first",
        })
        .reset_index()
        .rename(columns={"uthmani_token": "word_text", "root_ar": "root_ar_word"})
    )
    
    word_level = word_level.rename(columns={"root_ar_word": "root_ar"})
    
    return word_level


@st.cache_data
def create_verses(_word_level):
    """Create verse-level DataFrame from word-level."""
    # Verse text: join words
    verse_text = (
        _word_level.groupby(['chapter_id', 'verse_id'])['word_text']
        .apply(lambda x: ' '.join(x.astype(str)))
        .reset_index()
        .rename(columns={'word_text': 'verse_text'})
    )
    
    # Verse translation
    def join_trans(x):
        parts = x.dropna().astype(str)
        parts = parts[(parts.str.strip() != '_') & (parts.str.strip() != '')]
        return ' '.join(parts.unique())
    
    verse_trans = (
        _word_level.groupby(['chapter_id', 'verse_id'])['trans']
        .apply(join_trans)
        .reset_index()
        .rename(columns={'trans': 'verse_trans'})
    )
    
    # Merge
    verses = verse_text.merge(verse_trans, on=['chapter_id', 'verse_id'])
    verses['surah_name'] = verses['chapter_id'].map(SURAH_NAMES)
    verses['ref'] = verses.apply(
        lambda r: f"{r['surah_name']} ({r['chapter_id']}:{r['verse_id']})", axis=1
    )
    
    return verses


# ─────────────────────────────────────────────
#  SEMANTIC SEARCH — ChromaDB + silma model
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ تحميل نموذج التضمين...")
def load_semantic_model():
    """
    Load only the embedding model for query encoding.
    The document embeddings are already stored in ChromaDB.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner="📂 تحميل قاعدة التضمينات...")
def load_chroma_collections():
    """
    Load the pre-built ChromaDB collections from disk.
    Returns (verse_collection, tafsir_collection, error_message).
    Both collections may be None if ChromaDB or the path is unavailable.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collections = client.list_collections()
        col_names = [c.name for c in collections]

        verse_col  = client.get_collection(CHROMA_VERSE_COLLECTION)  if CHROMA_VERSE_COLLECTION  in col_names else None
        tafsir_col = client.get_collection(CHROMA_TAFSIR_COLLECTION) if CHROMA_TAFSIR_COLLECTION in col_names else None

        return verse_col, tafsir_col, None
    except Exception as e:
        return None, None, str(e)


def _encode_query(model, query: str) -> List[float]:
    """Encode query to a normalised embedding vector."""
    emb = model.encode([query], normalize_embeddings=True)
    return emb[0].tolist()


def _chroma_where_filter(surah_ids: Optional[List[int]]) -> Optional[dict]:
    """Build a ChromaDB $where filter for surah scoping."""
    if not surah_ids:
        return None
    if len(surah_ids) == 1:
        return {"chapter_id": {"$eq": surah_ids[0]}}
    return {"chapter_id": {"$in": surah_ids}}


def _cosine_from_distances(distances: List[float]) -> np.ndarray:
    """
    ChromaDB returns L2 distances by default when embeddings are normalised.
    For unit vectors: cosine_sim = 1 - (L2² / 2).
    If the collection was built with cosine space this is already a distance in [0,2].
    We convert to similarity in [0,1] either way.
    """
    d = np.array(distances, dtype=np.float32)
    # Clamp to avoid floating point issues
    return np.clip(1.0 - d / 2.0, 0.0, 1.0)

def make_ref(chapter_id: int, verse_id: int) -> str:
    return f"{int(chapter_id)}:{int(verse_id)}"
def safe_ref(meta):
    chapter_id = meta.get("chapter_id")
    verse_id   = meta.get("verse_id")

    # ✅ Best case: structured metadata exists
    if chapter_id is not None and verse_id is not None:
        return f"{int(chapter_id)}:{int(verse_id)}"

    # ⚠️ Fallback: try parsing existing ref
    ref = meta.get("ref")
    if ref and ":" in ref:
        try:
            c, v = map(int, ref.split(":"))
            return f"{c}:{v}"
        except:
            pass

    # ❌ Absolute fallback: invalid → skip later
    return None
def semantic_search(
    query: str,
    verse_col,
    tafsir_col,
    model,
    weight_verse: float,
    weight_tafsir: float,
    strategy: str,
    top_k: int,
    min_similarity: float,
    surah_ids: Optional[List[int]] = None,
    verses: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    if model is None:
        return pd.DataFrame()

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────
    def make_ref(chapter_id, verse_id):
        try:
            return f"{int(chapter_id)}:{int(verse_id)}"
        except:
            return None

    def safe_ref(meta):
        if not meta:
            return None

        # Preferred: structured IDs
        if meta.get("chapter_id") is not None and meta.get("verse_id") is not None:
            return make_ref(meta.get("chapter_id"), meta.get("verse_id"))

        # Fallback: parse ref
        ref = meta.get("ref")
        if ref and ":" in ref:
            try:
                c, v = map(int, ref.split(":"))
                return make_ref(c, v)
            except:
                return None

        return None

    # ─────────────────────────────────────────────
    # Encode query
    # ─────────────────────────────────────────────
    q_vec = _encode_query(model, query)
    where = _chroma_where_filter(surah_ids)

    fetch_k = min(top_k * 10, 6236)

    # ─────────────────────────────────────────────
    # Query verse collection
    # ─────────────────────────────────────────────
    verse_results = {}

    if verse_col is not None and strategy in ("verse", "weighted", "max"):
        try:
            kw = dict(
                query_embeddings=[q_vec],
                n_results=fetch_k,
                include=["metadatas", "distances"],
            )
            if where:
                kw["where"] = where

            res = verse_col.query(**kw)
            sims = _cosine_from_distances(res["distances"][0])

            for meta, sim in zip(res["metadatas"][0], sims):
                ref = safe_ref(meta)
                if not ref:
                    continue

                verse_results[ref] = {
                    "sim_v": float(sim),
                    "meta": meta,
                }

        except Exception as e:
            st.error(f"❌ خطأ في مجموعة الآيات: {str(e)}")
            return pd.DataFrame()

    # ─────────────────────────────────────────────
    # Query tafsir collection
    # ─────────────────────────────────────────────
    tafsir_results = {}

    if tafsir_col is not None and strategy in ("tafsir", "weighted", "max"):
        try:
            kw = dict(
                query_embeddings=[q_vec],
                n_results=fetch_k,
                include=["metadatas", "distances"],
            )
            if where:
                kw["where"] = where

            res = tafsir_col.query(**kw)
            sims = _cosine_from_distances(res["distances"][0])

            for meta, sim in zip(res["metadatas"][0], sims):
                ref = safe_ref(meta)
                if not ref:
                    continue

                tafsir_results[ref] = float(sim)

        except Exception as e:
            st.error(f"❌ خطأ في مجموعة التفسير: {str(e)}")
            return pd.DataFrame()

    # ─────────────────────────────────────────────
    # Merge + scoring
    # ─────────────────────────────────────────────
    all_refs = {
        r for r in set(verse_results) | set(tafsir_results)
        if r is not None
    }

    rows = []

    for ref in all_refs:
        try:
            chapter_id, verse_id = map(int, ref.split(":"))
        except:
            continue

        sim_v = verse_results.get(ref, {}).get("sim_v", 0.0)
        sim_t = tafsir_results.get(ref, 0.0)

        # ── scoring ───────────────────────────────
        if strategy == "verse":
            score = sim_v
        elif strategy == "tafsir":
            score = sim_t
        elif strategy == "weighted":
            w_v = weight_verse if sim_v > 0 else 0.0
            w_t = weight_tafsir if sim_t > 0 else 0.0
            total_w = w_v + w_t
            score = (sim_v * w_v + sim_t * w_t) / max(total_w, 1e-9)
        elif strategy == "max":
            score = max(sim_v, sim_t)
        else:
            score = sim_v

        if score < min_similarity:
            continue

        # ─────────────────────────────────────────
        # 🔥 SOURCE OF TRUTH: verses DataFrame
        # ─────────────────────────────────────────
        verse_text = ""
        verse_trans = ""

        if verses is not None:
            try:
                match = verses.loc[
                    (verses["chapter_id"].astype(int) == chapter_id) &
                    (verses["verse_id"].astype(int) == verse_id)
                ]
                if not match.empty:
                    verse_text = match.iloc[0]["verse_text"]
                    verse_trans = match.iloc[0]["verse_trans"]
            except:
                pass

        # optional tafsir text (only if present in verse meta)
        meta = verse_results.get(ref, {}).get("meta", {})
        tafsir_text = str(meta.get("tafsir", ""))[:300]

        rows.append({
            "ref": ref,
            "chapter_id": chapter_id,
            "verse_id": verse_id,
            "verse_text": verse_text,
            "verse_trans": verse_trans,
            "tafsir": tafsir_text,
            "sim_verse": round(sim_v, 4),
            "sim_tafsir": round(sim_t, 4),
            "final_score": round(score, 4),
        })

    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .sort_values("final_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    df.index += 1
    return df

# ─────────────────────────────────────────────
#  CLASSIC SEARCH FUNCTIONS (unchanged)
# ─────────────────────────────────────────────
def search_by_word(
    word_level: pd.DataFrame,
    verses: pd.DataFrame,
    query: str,
    match_mode: str = "any",
    exact: bool = False
) -> pd.DataFrame:
    """Search by word with optional exact matching."""
    words = [w.strip() for w in query.split() if w.strip()]
    if not words:
        return pd.DataFrame()
    
    norm_words = [normalize_arabic(w) for w in words]
    
    v = verses.copy()
    v["_vn"] = v["verse_text"].apply(normalize_arabic)
    
    def row_ok(vn: str) -> bool:
        if exact:
            toks = set(vn.split())
            chk = lambda w: w in toks
        else:
            chk = lambda w: w in vn
        
        if match_mode == "all":
            return all(chk(w) for w in norm_words)
        else:
            return any(chk(w) for w in norm_words)
    
    mask = v["_vn"].apply(row_ok)
    return v[mask].drop(columns=["_vn"]).reset_index(drop=True)


def search_by_lemma(
    word_level: pd.DataFrame,
    verses: pd.DataFrame,
    query: str,
    match_mode: str = "any"
) -> pd.DataFrame:
    """Search by lemma."""
    lemmas = [normalize_arabic(l.strip()) for l in query.split() if l.strip()]
    if not lemmas:
        return pd.DataFrame()
    
    wl = word_level.copy()
    wl["_ln"] = wl["lemma_ar"].apply(lambda x: normalize_arabic(x) if pd.notna(x) else "")
    
    hits = wl[wl["_ln"].isin(lemmas)]
    
    def verse_ok(g):
        found = set(g["_ln"].unique())
        if match_mode == "all":
            return all(l in found for l in lemmas)
        else:
            return any(l in found for l in lemmas)
    
    keys = (
        hits.groupby(["chapter_id", "verse_id"])
        .filter(verse_ok)[["chapter_id", "verse_id"]]
        .drop_duplicates()
    )
    
    return keys.merge(verses, on=["chapter_id", "verse_id"]).reset_index(drop=True)


def search_by_root(
    word_level: pd.DataFrame,
    verses: pd.DataFrame,
    query: str,
    match_mode: str = "any",
    use_pysarf_on_query: bool = True,
) -> pd.DataFrame:
    """Search by triliteral root."""
    terms = [r.strip() for r in str(query).split() if r.strip()]
    if not terms:
        return pd.DataFrame()
    
    if use_pysarf_on_query:
        norm_roots = [get_root_pysarf(t) for t in terms]
    else:
        norm_roots = [normalize_arabic(t) for t in terms]
    
    wl = word_level.copy()
    wl["_rn"] = wl["root_ar"].apply(lambda x: normalize_arabic(x) if pd.notna(x) else "")
    
    hits = wl[wl["_rn"].isin(norm_roots)]
    
    def verse_ok(g):
        found = set(g["_rn"].unique())
        if match_mode == "all":
            return all(r in found for r in norm_roots)
        return any(r in found for r in norm_roots)
    
    keys = (
        hits.groupby(["chapter_id", "verse_id"])
        .filter(verse_ok)[["chapter_id", "verse_id"]]
        .drop_duplicates()
    )
    
    return keys.merge(verses, on=["chapter_id", "verse_id"]).reset_index(drop=True)


# ─────────────────────────────────────────────
#  RENDERING
# ─────────────────────────────────────────────
def highlight_tokens(verse_text: str, query_tokens: list) -> str:
    """Wrap matched tokens with CSS highlight."""
    if not query_tokens:
        return verse_text
    norm_q = [normalize_arabic(qt) for qt in query_tokens]
    parts = []
    for tok in verse_text.split():
        if any(q in normalize_arabic(tok) for q in norm_q):
            parts.append(f'<span class="hl">{tok}</span>')
        else:
            parts.append(tok)
    return " ".join(parts)


def get_context_verse(verses, chapter_id: int, verse_id: int, offset: int) -> str:
    """Get context verse."""
    row = verses[
        (verses["chapter_id"] == chapter_id) &
        (verses["verse_id"] == verse_id + offset)
    ]
    return row.iloc[0]["verse_text"] if len(row) else ""


def _score_badge_html(row: pd.Series) -> str:
    """Render score badge for semantic results."""
    score_pct = int(row.get("final_score", 0) * 100)
    sim_v_pct = int(row.get("sim_verse", 0) * 100)
    sim_t_pct = int(row.get("sim_tafsir", 0) * 100)
    return (
        f'<div class="score-badge">'
        f'<span class="score-main">{score_pct}%</span>'
        f'<span class="score-detail">آية {sim_v_pct}% · تفسير {sim_t_pct}%</span>'
        f'</div>'
    )


def render_cards_html(page_results, query_tokens, show_context, show_trans, verses,
                      is_semantic: bool = False, show_tafsir: bool = False):
    """Render verse cards in HTML."""
    cards = ""
    for idx, row in page_results.iterrows():
        verse_hl = highlight_tokens(row["verse_text"], query_tokens)
        
        # Context verses (only meaningful for classic search)
        ctx_html = ""
        if show_context and not is_semantic:
            ctx_prev = get_context_verse(verses, row["chapter_id"], row["verse_id"], -1)
            ctx_next = get_context_verse(verses, row["chapter_id"], row["verse_id"], 1)
            if ctx_prev:
                ctx_html += f'<div class="ctx"><div class="ctx-label">آية سابقة</div><div class="ctx-text">{ctx_prev}</div></div>'
            ctx_html += f'<div class="ctx"><div class="ctx-label">الآية الحالية</div><div class="ctx-text">{verse_hl}</div></div>'
            if ctx_next:
                ctx_html += f'<div class="ctx"><div class="ctx-label">آية تالية</div><div class="ctx-text">{ctx_next}</div></div>'
        else:
            ctx_html = verse_hl
        
        trans_html  = f'<div class="verse-trans">{row.get("verse_trans","")}</div>' if show_trans else ""
        tafsir_html = ""
        if is_semantic and show_tafsir and row.get("tafsir"):
            tafsir_html = f'<div class="tafsir-text">📖 {str(row["tafsir"])[:280]}…</div>'
        score_html = _score_badge_html(row) if is_semantic else ""

        # ref may be enriched with surah name for classic mode
        ref_display = row.get("ref", f"{row['chapter_id']}:{row['verse_id']}")

        cards += f"""
        <div class="verse-card">
            <div class="card-header">
                <span class="verse-ref">{ref_display}</span>
                <div style="display:flex;align-items:center;gap:.5rem;">
                    {score_html}
                    <span class="verse-num">﴿{to_ar(int(row['verse_id']))}﴾</span>
                </div>
            </div>
            <div class="verse-text">{ctx_html}</div>
            {trans_html}
            {tafsir_html}
        </div>
        """
    
    full_html = f"""<!DOCTYPE html>
<html dir="rtl">
<head>
<meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;700&family=Amiri+Quran&family=Amiri:wght@400;700&display=swap');

@font-face {{
    font-family: 'KFGQPC';
    src: url('https://cdn.jsdelivr.net/gh/mustafa0x/qpc-fonts@master/fonts/UthmanicHafs1Ver18.ttf') format('truetype');
    font-display: swap;
}}

:root {{
    --bg2:       #111827;
    --bg3:       #1a2234;
    --border:    #2d3a4a;
    --gold:      #d4a017;
    --gold2:     #f0c040;
    --text:      #e8dcc8;
    --text2:     #c9b99a;
    --muted:     #6b7f8f;
    --hl-bg:     rgba(212,160,23,0.22);
    --hl-color:  #f0c040;
    --teal:      #4db6ac;
    --mushaf:    'KFGQPC','Amiri Quran','Noto Naskh Arabic','Amiri',serif;
    --ui:        'Amiri','Noto Naskh Arabic',serif;
    --r:         14px;
}}

* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
    background: transparent;
    font-family: var(--ui);
    color: var(--text);
    padding: 4px 0;
}}

.verse-card {{
    background: linear-gradient(150deg, var(--bg2) 0%, var(--bg3) 100%);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.4rem 1.8rem 1.1rem;
    margin-bottom: 1rem;
    box-shadow: 0 3px 18px rgba(0,0,0,.4);
    transition: border-color .25s, box-shadow .25s, transform .15s;
    position: relative;
    overflow: hidden;
}}

.verse-card:hover {{
    border-color: var(--gold);
    box-shadow: 0 8px 32px rgba(212,160,23,.1);
    transform: translateY(-2px);
}}

.card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: .8rem;
}}

.verse-ref {{
    font-size: .82rem;
    color: var(--gold);
    font-weight: 700;
    letter-spacing: .04em;
}}

.verse-num {{
    font-family: var(--mushaf);
    font-size: .9rem;
    color: #0b0f1a;
    background: var(--gold);
    border-radius: 50%;
    width: 2rem;
    height: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
}}

.verse-text {{
    font-family: var(--mushaf);
    font-size: 2.1rem;
    direction: rtl;
    text-align: justify;
    color: #f5ede0;
    line-height: 2.45;
    margin-bottom: .6rem;
    word-spacing: .1em;
}}

.hl {{
    background: var(--hl-bg);
    color: var(--hl-color);
    border-radius: 4px;
    padding: 0 3px;
    font-weight: 700;
}}

.ctx {{
    padding: .45rem .9rem;
    border-inline-end: 3px solid var(--border);
    margin: .35rem 0;
}}

.ctx-label {{
    font-size: .71rem;
    color: var(--muted);
    margin-bottom: .1rem;
}}

.ctx-text {{
    font-family: var(--mushaf);
    font-size: 1.1rem;
    direction: rtl;
    text-align: right;
    color: #3a5a4a;
    line-height: 1.9;
}}

.verse-trans {{
    font-size: .86rem;
    color: #6a8a7a;
    font-style: italic;
    line-height: 1.6;
    border-top: 1px solid var(--border);
    padding-top: .45rem;
    margin-top: .45rem;
}}

.tafsir-text {{
    font-size: .83rem;
    color: #8ab0c0;
    line-height: 1.7;
    border-top: 1px dashed #2a3a4a;
    padding-top: .45rem;
    margin-top: .4rem;
    direction: rtl;
}}

.score-badge {{
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 1px;
}}

.score-main {{
    font-size: .82rem;
    font-weight: 700;
    color: var(--teal);
    background: rgba(77,182,172,.12);
    border-radius: 8px;
    padding: 1px 7px;
}}

.score-detail {{
    font-size: .67rem;
    color: var(--muted);
}}
</style>
</head>
<body>
{cards}
</body>
</html>"""
    
    tafsir_extra = 60 if show_tafsir and is_semantic else 0
    card_height = 300 + (80 if show_context and not is_semantic else 0) + (40 if show_trans else 0) + tafsir_extra
    total_height = len(page_results) * card_height + 20
    components.html(full_html, height=total_height, scrolling=False)


def paginate(results: pd.DataFrame, page: int, page_size: int):
    """Paginate results."""
    total = len(results)
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    return results.iloc[start: start + page_size], page, total_pages


def inject_css():
    """Inject global CSS."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;700&family=Amiri+Quran&family=Amiri:wght@400;700&display=swap');

        @font-face {
            font-family: 'KFGQPC';
            src: url('https://cdn.jsdelivr.net/gh/mustafa0x/qpc-fonts@master/fonts/UthmanicHafs1Ver18.ttf') format('truetype');
            font-display: swap;
        }

        :root {
            --bg:         #0b0f1a;
            --bg2:        #111827;
            --bg3:        #1a2234;
            --border:     #2d3a4a;
            --gold:       #d4a017;
            --gold2:      #f0c040;
            --teal:       #4db6ac;
            --text:       #e8dcc8;
            --text2:      #c9b99a;
            --muted:      #6b7f8f;
            --r:          14px;
            --mushaf:     'KFGQPC', 'Amiri Quran', 'Noto Naskh Arabic', 'Amiri', serif;
            --ui:         'Amiri', 'Noto Naskh Arabic', serif;
        }

        html, body, [class*="css"] { font-family: var(--ui); }
        .stApp {
            background: linear-gradient(160deg, #07090f 0%, #0e1520 60%, #070910 100%);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #090e1a 0%, #0f1724 100%);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] * { color: var(--text2) !important; }
        [data-testid="stSidebar"] h3 { color: var(--gold) !important; }

        .app-header {
            text-align: center;
            padding: 2.2rem 1rem 1.6rem;
            background: radial-gradient(ellipse at top, #152015 0%, #091409 100%);
            border-radius: var(--r);
            border: 1px solid #2a4a2a;
            margin-bottom: 1.5rem;
        }

        .app-header .basmala {
            font-family: var(--mushaf);
            font-size: 2.9rem;
            color: var(--gold);
            direction: rtl;
            line-height: 2;
            margin-bottom: 0.4rem;
        }

        .app-header h1 {
            font-size: 1.5rem;
            color: var(--text2);
            margin: 0 0 0.25rem;
            font-weight: 400;
        }

        .app-header p { color: var(--muted); font-size: 0.88rem; margin: 0; }

        .stTextInput > div > div > input {
            background: var(--bg3) !important;
            color: var(--text) !important;
            border: 1.5px solid var(--border) !important;
            direction: rtl;
            font-family: var(--mushaf) !important;
            font-size: 1.25rem !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #1c3a1c, #2a502a) !important;
            color: var(--gold) !important;
            border: 1px solid #3a6a3a !important;
        }

        /* semantic mode button teal tint */
        .btn-semantic > button {
            background: linear-gradient(135deg, #0d2e2e, #1a4545) !important;
            color: var(--teal) !important;
            border: 1px solid #2a6a6a !important;
        }

        .stats-bar {
            display: flex;
            gap: .6rem;
            flex-wrap: wrap;
            margin-bottom: 1.1rem;
        }

        .chip {
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: .28rem .8rem;
            font-size: 0.79rem;
            color: var(--muted);
        }

        .chip b { color: var(--gold); }
        .chip.teal b { color: var(--teal); }

        .welcome {
            text-align: center;
            padding: 3rem 2rem;
        }

        .welcome .icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }

        .welcome .tagline {
            color: var(--text);
            font-size: 1.3rem;
            margin-bottom: 1rem;
        }

        .hints {
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
        }

        .hint-card {
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem;
            flex: 0 1 200px;
            font-size: 0.9rem;
        }

        .no-results {
            text-align: center;
            padding: 2rem;
            color: var(--muted);
            font-size: 1.2rem;
        }

        .page-info {
            text-align: center;
            padding: .5rem;
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        .sem-info {
            background: rgba(77,182,172,.07);
            border: 1px solid rgba(77,182,172,.2);
            border-radius: 10px;
            padding: .7rem 1rem;
            font-size: .82rem;
            color: #80cbc4;
            margin-bottom: .8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  SIDEBAR — SEMANTIC WEIGHT CONTROLS
# ─────────────────────────────────────────────
def render_semantic_sidebar():
    """Render the semantic-specific sidebar controls and return config dict."""
    st.markdown("### 🧠 إعدادات الدلالية")
    st.markdown("---")

    strategy = st.selectbox(
        "استراتيجية الدمج",
        options=["weighted", "verse", "tafsir", "max"],
        format_func=lambda s: {
            "weighted": "⚖️ مرجّح — Weighted (blend both signals)",
            "verse":    "📜 آية فقط — Verse only",
            "tafsir":   "📖 تفسير فقط — Tafsir only",
            "max":      "🔝 الأقصى — Max (best of both)",
        }[s],
        index=0,
        help="اختر كيفية دمج إشارات الآية والتفسير",
    )

    w_verse = w_tafsir = 0.5
    if strategy == "weighted":
        st.markdown("**الأوزان** _(لا تحتاج إعادة تضمين)_")
        st.info("🔄 تعديل الأوزان أدناه لا يتطلب إعادة حساب التضمينات — التأثير فوري!")
        w_verse  = st.slider(
            "وزن الآية",    
            0.0, 1.0, 0.40, 0.05, 
            key="w_verse",
            help="كم نسبة اعتماد نتائج البحث على نص الآية ذاته"
        )
        w_tafsir = st.slider(
            "وزن التفسير",  
            0.0, 1.0, 0.60, 0.05, 
            key="w_tafsir",
            help="كم نسبة اعتماد نتائج البحث على تفسير الآية"
        )

    st.markdown("---")
    top_k = st.slider(
        "عدد النتائج top-k", 
        3, 50, 10, 1, 
        key="sem_topk",
        help="كم عدد أفضل النتائج المراد عرضها"
    )
    min_sim = st.slider(
        "حد الدرجة الأدنى", 
        0.0, 0.9, 0.20, 0.01, 
        key="sem_minsim",
        help="تجاهل النتائج التي تقل درجتها عن هذا الحد"
    )

    show_tafsir_cards = st.checkbox(
        "عرض مقتطف التفسير", 
        value=True, 
        key="show_tafsir",
        help="إظهار أول 280 حرف من التفسير تحت كل آية"
    )

    return dict(
        strategy=strategy,
        weight_verse=w_verse,
        weight_tafsir=w_tafsir,
        top_k=top_k,
        min_similarity=min_sim,
        show_tafsir_cards=show_tafsir_cards,
    )


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
def main():
    inject_css()

    # ── Load base data ────────────────────────────────────────────────
    with st.spinner("📦 Loading data..."):
        df_linguistic, _, _ = load_data()
        word_level = process_word_level(df_linguistic)
        verses = create_verses(word_level)

    # App header
    st.markdown(
        """
        <div class="app-header">
            <div class="basmala">بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ</div>
            <h1>🔍 البحث في القرآن الكريم</h1>
            <p>Quranic Search System — root · word · lemma · semantic (silma-ai)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ إعدادات البحث")
        st.markdown("---")

        search_type = st.radio(
            "نوع البحث",
            ["🌱 جذر | Root", "📖 كلمة | Word", "📚 لمّة | Lemma", "🧠 دلالي | Semantic"],
            index=0,
        )

        is_semantic = "Semantic" in search_type

        # ── Semantic controls ──────────────────────────────────────
        sem_cfg = {}
        if is_semantic:
            sem_cfg = render_semantic_sidebar()
        else:
            match_mode = st.radio(
                "وضع المطابقة",
                ["أي كلمة — OR", "كل الكلمات — AND"],
                index=0,
            )
            match_mode_val = "all" if "AND" in match_mode else "any"

            exact_word = False
            if "Word" in search_type:
                exact_word = st.checkbox("مطابقة تامة للكلمة", value=False)

            st.markdown("---")
            st.markdown("### 🎨 خيارات العرض")
            show_context = st.checkbox("عرض السياق", value=False)

        st.markdown("---" if not is_semantic else "")
        show_trans = st.checkbox("عرض الترجمة الإنجليزية", value=True, key="show_trans_global")

        st.markdown("---")
        st.markdown("### 🔎 تصفية السور")
        surah_options = ["الكل"] + [f"{k} — {v}" for k, v in SURAH_NAMES.items()]
        selected_surah = st.selectbox("اختر سورة", surah_options, index=0)

        # Multi-surah selection for semantic mode
        selected_surahs_multi = []
        if is_semantic:
            use_multi = st.checkbox("تحديد عدة سور", value=False, key="sem_multi")
            if use_multi:
                selected_surahs_multi = st.multiselect(
                    "اختر السور",
                    options=list(SURAH_NAMES.keys()),
                    format_func=lambda k: f"{k} — {SURAH_NAMES[k]}",
                    key="sem_surahs_multi",
                )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:.72rem;color:#2a4a3a;text-align:center;'>"
            "٧٧٬٤٢٩ كلمة · ٦٬٢٣٦ آية · ١١٤ سورة"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Search bar ────────────────────────────────────────────────────
    placeholders = {
        "🌱 جذر | Root":       "أدخل جذرًا… مثال: رحم",
        "📖 كلمة | Word":      "أدخل كلمة… مثال: الرحمن",
        "📚 لمّة | Lemma":     "أدخل لمّة… مثال: رحيم",
        "🧠 دلالي | Semantic": "أدخل سؤالاً أو جملة… مثال: الصبر على الابتلاء",
    }

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input(
            "q",
            placeholder=placeholders.get(search_type, ""),
            label_visibility="collapsed",
            key="q",
        )
    with col_btn:
        btn_container = st.container()
        with btn_container:
            search_clicked = st.button("🔍 بحث", use_container_width=True)

    # ── Session state ─────────────────────────────────────────────────
    if "page" not in st.session_state:
        st.session_state.page = 1

    q_key = f"{query}|{search_type}|{selected_surah}"
    if st.session_state.get("_qkey") != q_key:
        st.session_state.page = 1
        st.session_state._qkey = q_key

    # ── Execute search ────────────────────────────────────────────────
    if search_clicked or query.strip():
        if not query.strip():
            st.warning("الرجاء إدخال نص للبحث.")
            return

        # ── Determine surah scope ─────────────────────────────────
        fv  = verses.copy()
        fwl = word_level.copy()
        scope_label = "كل السور"
        surah_ids = None  # For semantic filtering

        if is_semantic:
            # Build surah_ids list for ChromaDB filtering
            if selected_surahs_multi:
                surah_ids = selected_surahs_multi
                scope_label = " + ".join(SURAH_NAMES[s] for s in surah_ids[:3])
                if len(surah_ids) > 3:
                    scope_label += f" +{len(surah_ids)-3}"
            elif selected_surah != "الكل":
                try:
                    sid = int(selected_surah.split(" — ")[0])
                    surah_ids = [sid]
                    scope_label = selected_surah
                except (ValueError, IndexError):
                    surah_ids = None
                    scope_label = "كل السور"
            else:
                surah_ids = None   # full Quran
        else:
            # Classic search: filter DataFrames directly
            if selected_surah != "الكل":
                try:
                    sid = int(selected_surah.split(" — ")[0])
                    fv  = fv[fv["chapter_id"] == sid]
                    fwl = fwl[fwl["chapter_id"] == sid]
                    scope_label = selected_surah
                except (ValueError, IndexError):
                    scope_label = "كل السور"

        # ── Run search ────────────────────────────────────────────
        with st.spinner("جاري البحث…"):
            if is_semantic:
                # Load ChromaDB + model (cached)
                verse_col, tafsir_col, chroma_err = load_chroma_collections()
                sem_model, model_err = load_semantic_model()

                if chroma_err or (verse_col is None and tafsir_col is None):
                    st.error(
                        f"❌ لم يتم العثور على قاعدة التضمينات المحفوظة\n\n"
                        f"**المسار المتوقع:** `{CHROMA_PATH.resolve()}`\n\n"
                        f"**الحل:**\n"
                        f"1. تأكد من تشغيل جميع خلايا 'Semantic Search' في دفتر Classical.ipynb أولاً\n"
                        f"2. سيتم إنشاء مجلدات ChromaDB تلقائياً مع حفظ التضمينات\n"
                        f"3. بعدها ستكون جاهزة للاستخدام في هذا التطبيق\n\n"
                        f"**التفاصيل التقنية:** {chroma_err or 'لم يتم العثور على مجموعات'}"
                    )
                    return
                if model_err:
                    st.error(
                        f"❌ فشل تحميل نموذج التضمين\n\n"
                        f"**النموذج:** {SEMANTIC_MODEL_NAME}\n\n"
                        f"**الخطأ:** {model_err}\n\n"
                        f"**الحل:** تأكد من اتصالك بالإنترنت وتوفر مساحة كافية"
                    )
                    return

                if sem_model is None:
                    st.error("❌ فشل في تحميل نموذج التضمين")
                    return

                results = semantic_search(
                    query=query,
                    verse_col=verse_col,
                    tafsir_col=tafsir_col,
                    model=sem_model,
                    weight_verse=sem_cfg.get("weight_verse", 0.40),
                    weight_tafsir=sem_cfg.get("weight_tafsir", 0.60),
                    strategy=sem_cfg.get("strategy", "weighted"),
                    top_k=sem_cfg.get("top_k", 10),
                    min_similarity=sem_cfg.get("min_similarity", 0.20),
                    surah_ids=surah_ids if is_semantic else None,
                    verses=verses if is_semantic else None, #ADDED
                )
                query_tokens = []   # semantic — no token highlighting

            elif "Root" in search_type:
                results = search_by_root(fwl, fv, query, match_mode_val)
                query_tokens = query.split()
            elif "Word" in search_type:
                results = search_by_word(fwl, fv, query, match_mode_val, exact_word)
                query_tokens = query.split()
            else:
                results = search_by_lemma(fwl, fv, query, match_mode_val)
                query_tokens = query.split()

        total = len(results)

        # ── Stats bar ─────────────────────────────────────────────
        chip_class = "chip teal" if is_semantic else "chip"
        st.markdown(
            f'<div class="stats-bar">'
            f'<div class="{chip_class}">📊 النتائج: <b>{total}</b> آية</div>'
            f'<div class="{chip_class}">🔍 <b>{query.strip()}</b></div>'
            f'<div class="{chip_class}">نوع: <b>{search_type.split("|")[0].strip()}</b></div>'
            f'<div class="{chip_class}">نطاق: <b>{scope_label}</b></div>'
            + (f'<div class="{chip_class}">استراتيجية: <b>{sem_cfg.get("strategy","")}</b></div>' if is_semantic else '')
            + f'</div>',
            unsafe_allow_html=True,
        )

        if is_semantic:
            st.markdown(
                f'<div class="sem-info">🤖 النموذج: <b>{SEMANTIC_MODEL_NAME}</b> · '
                f'التضمينات محفوظة مسبقاً في ChromaDB · الأوزان قابلة للتعديل دون إعادة تضمين</div>',
                unsafe_allow_html=True,
            )

        if total == 0:
            st.markdown(
                '<div class="no-results">😔 لا توجد نتائج مطابقة</div>',
                unsafe_allow_html=True,
            )
            return

        # ── Paginate ──────────────────────────────────────────────
        page_results, cur_page, total_pages = paginate(results, st.session_state.page, PAGE_SIZE)

        st.markdown(
            f'<div class="page-info">الصفحة <b>{to_ar(cur_page)}</b> / <b>{to_ar(total_pages)}</b></div>',
            unsafe_allow_html=True,
        )

        render_cards_html(
            page_results,
            query_tokens,
            show_context=(not is_semantic and show_context),
            show_trans=show_trans,
            verses=verses,
            is_semantic=is_semantic,
            show_tafsir=sem_cfg.get("show_tafsir_cards", False) if is_semantic else False,
        )

        # ── Pagination controls ───────────────────────────────────
        if total_pages > 1:
            pc = st.columns([1, 1, 2, 1, 1])
            with pc[0]:
                if st.button("⏮ أول", disabled=(cur_page == 1), key="p_first"):
                    st.session_state.page = 1
                    st.rerun()
            with pc[1]:
                if st.button("◀ سابق", disabled=(cur_page == 1), key="p_prev"):
                    st.session_state.page = cur_page - 1
                    st.rerun()
            with pc[2]:
                st.markdown(
                    f'<div style="text-align:center;padding:.35rem;color:#d4a017;font-weight:700;">'
                    f'{to_ar(cur_page)} / {to_ar(total_pages)}</div>',
                    unsafe_allow_html=True,
                )
            with pc[3]:
                if st.button("التالي ▶", disabled=(cur_page == total_pages), key="p_next"):
                    st.session_state.page = cur_page + 1
                    st.rerun()
            with pc[4]:
                if st.button("آخر ⏭", disabled=(cur_page == total_pages), key="p_last"):
                    st.session_state.page = total_pages
                    st.rerun()

    else:
        # Welcome screen
        st.markdown(
            """
            <div class="welcome">
                <div class="icon">📖</div>
                <div class="tagline">ابدأ بكتابة جذر أو كلمة أو سؤال للبحث</div>
                <hr style="border-color:#1a2a1a;margin:1.5rem auto;width:35%"/>
                <div class="hints">
                    <div class="hint-card">
                        <b>🌱 جذر</b><br>
                        رحم · حمد · علم
                    </div>
                    <div class="hint-card">
                        <b>📖 كلمة</b><br>
                        الرحمن · ربّ
                    </div>
                    <div class="hint-card">
                        <b>📚 لمّة</b><br>
                        رَحِيم · عَالَم
                    </div>
                    <div class="hint-card">
                        <b>🧠 دلالي</b><br>
                        الصبر على الابتلاء
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
