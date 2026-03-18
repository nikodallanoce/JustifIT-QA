import itertools
from typing import Iterable, List, Dict, Optional, Union, Any
from dataclasses import dataclass
import os
from datasets import Dataset
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter


# -------------------------------
# Utilities
# -------------------------------
def default_text_splitter(chunk_size: int = 1500, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """Default text splitter for legal texts."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# -------------------------------
# Config dataclass
# -------------------------------
@dataclass
class BuildConfig:
    text_field: str = "content"  # dataset column containing the text
    id_field: Optional[str] = None  # optional ID field
    extra_meta_fields: Optional[List[str]] = None  # other fields to keep as metadata


# -------------------------------
# Core class
# -------------------------------
class LegalFAISSRetriever:
    """
    Build and query a FAISS vector index (via LangChain) over a legal corpus.

    Parameters
    ----------
    corpus : HuggingFace Dataset or iterable of dicts
        Each element must include at least the field defined in config.text_field.
    embedding : Embeddings
        A *loaded* LangChain embedding model instance (e.g. SentenceTransformerEmbeddings).
    splitter : RecursiveCharacterTextSplitter
        The text splitter used for chunking.
    config : BuildConfig
        Defines how to access text/id/metadata fields.
    """

    def __init__(
            self,
            corpus: Union[Dataset, Iterable[Dict[str, Any]]],
            embedding: Optional[Embeddings],
            splitter: Optional[TextSplitter] = None,
            config: Optional[BuildConfig] = None,
    ):
        self.corpus = corpus
        self.embedding = embedding
        self.splitter = splitter  # or default_text_splitter()
        self.config = config or BuildConfig()
        self._faiss: Optional[FAISS] = None

    # ---------- Build ----------
    def build(self, chunked_docs: Optional[List[Document]] = None) -> "LegalFAISSRetriever":
        """Chunk corpus and build FAISS index."""
        docs = self.to_documents(self.corpus) if chunked_docs is None else chunked_docs
        if not docs:
            raise ValueError("No documents created — check text_field, chunking, and corpus content.")
        self._faiss: FAISS = FAISS.from_documents(docs, self.embedding,
                                                  normalize_L2=True)  # distance_strategy=DistanceStrategy.COSINE)
        return self

    def to_documents(self, corpus: Union[Dataset, Iterable[Dict[str, Any]]]) -> List[Document]:
        text_f = self.config.text_field
        id_f = self.config.id_field
        extras = set(self.config.extra_meta_fields or [])
        docs: List[Document] = []

        for i, row in enumerate(corpus):
            text = row.get(text_f) if isinstance(row, dict) else row[text_f]
            if not isinstance(text, str) or not text.strip():
                continue

            doc_id = (
                str(row.get(id_f))
                if id_f and isinstance(row, dict)
                else (str(row[id_f]) if id_f and id_f in row else f"doc_{i}")
            )

            md: Dict[str, Any] = {"doc_id": doc_id}
            for k in extras:
                val = row.get(k) if isinstance(row, dict) else row[k]
                if val is not None:
                    md[k] = val

            chunks = [text]
            if self.splitter is not None:
                chunks = self.splitter.split_text(text)
            for cidx, chunk in enumerate(chunks):
                docs.append(Document(page_content=chunk, metadata={**md, "chunk_id": cidx}))
        return docs

    # ---------- Retrieval ----------
    def retrieve(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        """Return top-k chunks as dicts with rank, score, text, and metadata."""
        if self._faiss is None:
            raise RuntimeError("Index not built. Call .build() or .load() first.")
        results = self._faiss.similarity_search_with_relevance_scores(query, k=k)
        return [
            {
                "rank": i + 1,
                "score": float(score),
                "text": doc.page_content,
                "metadata": dict(doc.metadata),
            }
            for i, (doc, score) in enumerate(results)
        ]

    def retrieve_text_only(self, query: str, k: int = 8) -> List[str]:
        results = self.retrieve(query, k=k)
        return [r['text'] for r in results]

    # ---------- Persistence ----------
    def save(self, path: str) -> None:
        """Save FAISS index and store to a directory."""
        if self._faiss is None:
            raise RuntimeError("Nothing to save — build or load an index first.")
        os.makedirs(path, exist_ok=True)
        self._faiss.save_local(path)

    def load(self, path: str) -> "LegalFAISSRetriever":
        """Load a previously saved FAISS index from directory."""
        if not os.path.isdir(path):
            raise FileNotFoundError(f"No directory found at: {path}")
        self._faiss = FAISS.load_local(path, self.embedding, allow_dangerous_deserialization=True)
        return self
