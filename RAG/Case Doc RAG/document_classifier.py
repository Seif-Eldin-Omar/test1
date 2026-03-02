"""
document_classifier.py

Classifies Egyptian civil-case documents using a two-stage approach:
1. Heuristic keyword matching on the header and body excerpt.
2. LLM-based semantic classification as a fallback when heuristic
   confidence is below the threshold.

Converted from document_classifier.ipynb and cleaned up for production use.

Usage:
    from document_classifier import classify_document

    result = classify_document("صحيفة دعوى مقدمة من المدعي...")
    print(result["final_type"])       # e.g. "صحيفة دعوى"
    print(result["confidence"])       # e.g. 75
    print(result["explanation"])      # reasoning
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ClassifierState(TypedDict):
    text: str
    header: str
    body_excerpt: str

    heuristic_type: Optional[str]
    heuristic_confidence: int
    matched_keywords: List[str]

    final_type: Optional[str]
    confidence: int
    explanation: Optional[str]


# ---------------------------------------------------------------------------
# Keyword dictionary for heuristic matching
# ---------------------------------------------------------------------------

KEYWORDS: Dict[str, List[str]] = {
    "صحيفة دعوى": ["صحيفة دعوى", "الطلبات", "الوقائع", "بناءً عليه"],
    "مذكرة بدفاع": ["مذكرة بدفاع", "الدفاع", "أولاً", "ثانياً"],
    "حكم": ["باسم الشعب", "فلهذه الأسباب", "قضت المحكمة", "وحيث إن"],
    "محضر جلسة": ["انعقدت المحكمة", "أثبت الحاضرون", "قررت المحكمة"],
    "إعلان": ["إنه في يوم", "أعلنت", "كلفته الحضور"],
    "أمر أداء": ["أمر أداء", "يأمر", "دين ثابت"],
    "أمر على عريضة": ["أمر على عريضة", "بعد الاطلاع", "نأمر"],
    "تقرير خبير": ["تقرير الخبير", "المعاينة", "الخبير المنتدب"],
    "محضر إثبات حالة": ["إثبات حالة", "قام المحضر", "ثبت لديه"],
    "مستند غير معروف": [],
}

# Confidence threshold -- below this the LLM fallback is used
HEURISTIC_CONFIDENCE_THRESHOLD = 70

# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def extract_header_node(state: ClassifierState) -> ClassifierState:
    """Split the document into a header (first 6 lines) and a body excerpt."""
    lines = state["text"].split("\n")
    header = "\n".join(lines[:6])
    body_excerpt = " ".join(state["text"].split()[:400])

    state["header"] = header
    state["body_excerpt"] = body_excerpt
    return state


def heuristic_node(state: ClassifierState) -> ClassifierState:
    """Score each document type by counting keyword matches."""
    text = state["header"] + "\n" + state["body_excerpt"]
    best_type: Optional[str] = None
    best_score = 0
    best_keywords: List[str] = []

    for doc_type, keys in KEYWORDS.items():
        matches = [k for k in keys if k in text]
        score = len(matches) * 25  # each match adds 25 points

        if score > best_score:
            best_score = score
            best_type = doc_type
            best_keywords = matches

    state["heuristic_type"] = best_type
    state["heuristic_confidence"] = min(best_score, 100)
    state["matched_keywords"] = best_keywords
    state["explanation"] = "Heuristic-based classification"

    return state


def check_confidence_node(state: ClassifierState) -> ClassifierState:
    """Pass-through node used as a routing checkpoint."""
    return state


def confidence_router(state: ClassifierState) -> str:
    """Route to heuristic result or LLM fallback based on confidence."""
    if state["heuristic_confidence"] >= HEURISTIC_CONFIDENCE_THRESHOLD:
        state["final_type"] = state["heuristic_type"]
        state["confidence"] = state["heuristic_confidence"]
        state["explanation"] = (
            "تم تحديد النوع بناءً على الكلمات المفتاحية: "
            + ", ".join(state["matched_keywords"])
        )
        return "use_heuristic"
    return "use_llm"


def llm_classifier_node(state: ClassifierState) -> ClassifierState:
    """Use an LLM to classify the document when heuristics are not confident."""
    llm_model = os.getenv("CLASSIFIER_LLM_MODEL", "gemini-2.0-flash-lite")
    llm = ChatGoogleGenerativeAI(model=llm_model)

    prompt = f"""
    صنّف نوع هذا المستند المدني المصري بدقة شديدة.
    النص:
    {state["header"]}

    {state["body_excerpt"]}

    الأنواع المحتملة:
    {list(KEYWORDS.keys())}

    أرجع JSON فقط بالشكل:
    {{"doc_type": "...", "confidence": 0-100, "reasons": "..."}}
    """

    try:
        result = llm.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)

        # Try to parse JSON from the response
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        parsed = json.loads(content)
        state["final_type"] = parsed.get("doc_type", "مستند غير معروف")
        state["confidence"] = parsed.get("confidence", 80)
        state["explanation"] = parsed.get("reasons", "LLM-based classification")
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("LLM classifier failed to parse response: %s", exc)
        state["final_type"] = state.get("heuristic_type") or "مستند غير معروف"
        state["confidence"] = max(state.get("heuristic_confidence", 0), 30)
        state["explanation"] = "LLM classification failed; fell back to heuristic"

    return state


# ---------------------------------------------------------------------------
# Build and compile the classifier graph
# ---------------------------------------------------------------------------

def _build_classifier_graph():
    """Construct and compile the document classifier LangGraph workflow."""
    graph = StateGraph(ClassifierState)

    graph.add_node("extract_header", extract_header_node)
    graph.add_node("heuristic", heuristic_node)
    graph.add_node("check_confidence", check_confidence_node)
    graph.add_node("llm_classifier", llm_classifier_node)

    graph.set_entry_point("extract_header")

    graph.add_edge("extract_header", "heuristic")
    graph.add_edge("heuristic", "check_confidence")

    graph.add_conditional_edges(
        "check_confidence",
        confidence_router,
        {
            "use_heuristic": END,
            "use_llm": "llm_classifier",
        },
    )

    graph.add_edge("llm_classifier", END)

    return graph.compile()


# Compiled classifier graph
_classifier_app = _build_classifier_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_document(text: str) -> Dict[str, Any]:
    """Classify a single document text and return its type, confidence, and explanation.

    Parameters
    ----------
    text:
        The raw text content of the document (post-OCR or plain text).

    Returns
    -------
    dict with keys:
        - final_type: str -- the classified document type
        - confidence: int -- confidence score (0-100)
        - explanation: str -- reasoning for the classification
    """
    if not text or not text.strip():
        return {
            "final_type": "مستند غير معروف",
            "confidence": 0,
            "explanation": "Empty document text",
        }

    initial_state: ClassifierState = {
        "text": text,
        "header": "",
        "body_excerpt": "",
        "heuristic_type": None,
        "heuristic_confidence": 0,
        "matched_keywords": [],
        "final_type": None,
        "confidence": 0,
        "explanation": None,
    }

    result = _classifier_app.invoke(initial_state)

    return {
        "final_type": result.get("final_type", "مستند غير معروف"),
        "confidence": result.get("confidence", 0),
        "explanation": result.get("explanation", ""),
    }
