from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from .config import settings


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Create an Ollama-backed chat model."""
    return ChatOllama(
        model=model or settings.ollama_model,
        temperature=temperature,
    )


def get_embeddings(model: Optional[str] = None) -> OllamaEmbeddings:
    """Create an Ollama-based embeddings model."""
    return OllamaEmbeddings(
        model=model or settings.embedding_model,
    )


def build_rag_chain(retriever, llm: Optional[BaseChatModel] = None):
    """Return a simple callable that runs retrieval + generation.

    This avoids depending on `langchain.chains` so it works across LangChain
    versions that reorganize chain modules.
    """
    if llm is None:
        llm = get_llm()

    system_prompt = (
        "You are a helpful assistant answering questions based on the provided context.\n"
        "Use ONLY the information in the context to answer.\n"
        "If the answer cannot be found in the context, say you don't know.\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Provide a concise, well-structured answer.",
            ),
        ]
    )

    def _run(question: str):
        # Retrieve relevant documents (support both old and new retriever APIs)
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(question)
        else:
            # New-style retrievers are Runnables and use `.invoke`
            docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        # Build the chat prompt and invoke the model
        messages = prompt.format_messages(context=context, question=question)
        response = llm.invoke(messages)

        # ChatOllama returns a Message with `.content`
        answer_text = getattr(response, "content", str(response))

        return {
            "result": answer_text,
            "source_documents": docs,
        }

    return _run

