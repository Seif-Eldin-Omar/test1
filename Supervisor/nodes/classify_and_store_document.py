"""
classify_and_store_document.py

Supervisor node that classifies uploaded documents (after OCR or directly)
and stores them in MongoDB with the correct document type as their title.

This node runs after dispatch_agents when OCR was involved, or processes
raw text documents directly. It ensures every document stored in the
database has a proper classified name so the document selector in
rag_docs.py can find them reliably.
"""

import logging
import os
import sys
from typing import Any, Dict, List

from pymongo import MongoClient

from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)


def _get_classifier():
    """Lazy-import the document classifier to avoid circular imports."""
    classifier_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "RAG", "Case Doc RAG",
    )
    classifier_dir = os.path.normpath(classifier_dir)
    if classifier_dir not in sys.path:
        sys.path.insert(0, classifier_dir)

    from document_classifier import classify_document
    return classify_document


# Module-level MongoDB client (reused across calls to avoid connection leaks)
_mongo_client = None

def _get_mongo_collection():
    """Get the MongoDB collection used for document storage."""
    global _mongo_client
    if _mongo_client is None:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        _mongo_client = MongoClient(mongo_uri)
    db = _mongo_client["Rag"]
    return db["Document Storage"]


def classify_and_store_document_node(state: SupervisorState) -> Dict[str, Any]:
    """Classify each document from OCR output (or raw uploads) and store it.

    This node inspects ``agent_results`` for OCR output. If OCR was run,
    it takes the extracted text, classifies the document type, and stores
    it in MongoDB with the classified type as the title.

    If no OCR was needed but raw text files were uploaded, it reads them
    directly, classifies, and stores.

    Updates state keys: ``document_classifications``.
    """
    agent_results = state.get("agent_results", {})
    uploaded_files = state.get("uploaded_files", [])
    case_id = state.get("case_id", "")

    classify_document = _get_classifier()
    collection = _get_mongo_collection()

    classifications: List[Dict[str, Any]] = []

    # Case 1: OCR was run -- classify the extracted text
    ocr_result = agent_results.get("ocr")
    if ocr_result:
        raw_texts = ocr_result.get("raw_output", {}).get("raw_texts", [])
        if not raw_texts:
            # Fall back to the combined response
            combined = ocr_result.get("response", "")
            if combined:
                raw_texts = [combined]

        for i, text in enumerate(raw_texts):
            if not text or not text.strip():
                logger.warning("Skipping empty OCR text at index %d", i)
                continue

            result = classify_document(text)
            doc_type = result.get("final_type", "مستند غير معروف")
            confidence = result.get("confidence", 0)
            explanation = result.get("explanation", "")

            # Build a descriptive title from the type and case
            file_ref = uploaded_files[i] if i < len(uploaded_files) else f"doc_{i}"
            title = doc_type

            # Store in MongoDB
            doc_record = {
                "title": title,
                "doc_type": doc_type,
                "case_id": case_id,
                "source_file": file_ref,
                "text": text,
                "classification_confidence": confidence,
                "classification_explanation": explanation,
            }

            try:
                insert_result = collection.insert_one(doc_record)
                logger.info(
                    "Stored document: title='%s', type='%s', id=%s",
                    title, doc_type, insert_result.inserted_id,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to store document '%s': %s", title, exc
                )

            classifications.append({
                "file": file_ref,
                "title": title,
                "doc_type": doc_type,
                "confidence": confidence,
                "explanation": explanation,
            })

        logger.info(
            "Classified and stored %d document(s) from OCR output",
            len(classifications),
        )

    # Case 2: No OCR, but text files were uploaded directly
    elif uploaded_files:
        for file_path in uploaded_files:
            try:
                if not os.path.isfile(file_path):
                    logger.warning("File not found: %s", file_path)
                    continue

                # Only process text files
                if not file_path.endswith((".txt", ".text")):
                    logger.info(
                        "Skipping non-text file (needs OCR): %s", file_path
                    )
                    continue

                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                if not text.strip():
                    continue

                result = classify_document(text)
                doc_type = result.get("final_type", "مستند غير معروف")
                confidence = result.get("confidence", 0)
                explanation = result.get("explanation", "")

                title = doc_type

                doc_record = {
                    "title": title,
                    "doc_type": doc_type,
                    "case_id": case_id,
                    "source_file": file_path,
                    "text": text,
                    "classification_confidence": confidence,
                    "classification_explanation": explanation,
                }

                try:
                    insert_result = collection.insert_one(doc_record)
                    logger.info(
                        "Stored document: title='%s', type='%s', id=%s",
                        title, doc_type, insert_result.inserted_id,
                    )
                except Exception as exc:
                    logger.exception(
                        "Failed to store document '%s': %s", title, exc
                    )

                classifications.append({
                    "file": file_path,
                    "title": title,
                    "doc_type": doc_type,
                    "confidence": confidence,
                    "explanation": explanation,
                })

            except Exception as exc:
                logger.exception(
                    "Failed to process file '%s': %s", file_path, exc
                )

        logger.info(
            "Classified and stored %d text document(s) directly",
            len(classifications),
        )

    return {
        "document_classifications": classifications,
    }
