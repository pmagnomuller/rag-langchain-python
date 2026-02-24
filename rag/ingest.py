from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import settings


def _iter_files(root: Path) -> Iterable[Path]:
    """Yield supported files under the given root directory."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".txt", ".md", ".markdown", ".pdf"}:
            yield path


def load_documents(data_dir: Path | str | None = None) -> List[Document]:
    """Load documents from the data directory into LangChain `Document`s."""
    base = Path(data_dir) if data_dir is not None else settings.data_dir
    base.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []

    for path in _iter_files(base):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")

        docs.extend(loader.load())

    return docs


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(documents))


def build_vector_store(
    chunks: Sequence[Document],
    embeddings: Embeddings,
    persist_directory: Path | str | None = None,
) -> Chroma:
    """Create (or overwrite) a Chroma vector store from document chunks."""
    persist_path = Path(persist_directory) if persist_directory is not None else settings.chroma_dir
    persist_path.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=list(chunks),
        embedding=embeddings,
        persist_directory=str(persist_path),
    )
    return vector_store


def get_retriever(
    embeddings: Embeddings,
    persist_directory: Path | str | None = None,
    k: int = 4,
):
    """Return a retriever backed by the (persisted) Chroma vector store.

    Assumes `build_vector_store` has already been called at least once.
    """
    persist_path = Path(persist_directory) if persist_directory is not None else settings.chroma_dir
    persist_path.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    class RetrieverAdapter:
        """Adapter that supports both old and new retriever APIs."""

        def __init__(self, inner):
            self._inner = inner

        def get_relevant_documents(self, query):
            if hasattr(self._inner, "get_relevant_documents"):
                return self._inner.get_relevant_documents(query)
            return self._inner.invoke(query)

        def invoke(self, query, *args, **kwargs):
            if hasattr(self._inner, "invoke"):
                return self._inner.invoke(query, *args, **kwargs)
            return self._inner.get_relevant_documents(query)

        def __getattr__(self, name):
            # Delegate any other attributes to the underlying retriever
            return getattr(self._inner, name)

    return RetrieverAdapter(base_retriever)

