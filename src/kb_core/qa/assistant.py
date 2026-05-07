"""RAG-powered Q&A assistant.

Uses the LLM abstraction (kb_core.llm), so model id picks the provider:
'gemini-*' / 'deepseek-*' / 'gpt-*' / etc.

Domain expertise is supplied per-project via DomainContext. The
guidelines below are framework-generic; the role description
("You are an expert in ...") comes from the project config.
"""

from __future__ import annotations

import textwrap

from ..config import DomainContext
from ..knowledge_base import KnowledgeBase
from ..llm import LLMBackend, get_llm


_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""
{role_description}

Guidelines:
1. Be precise and cite the source papers/sections when you use retrieved context.
2. Distinguish clearly between information from the provided papers vs. general knowledge.
3. When recommending parameters, state the rationale and the trade-offs.
4. If the question cannot be answered from available context, say so clearly and
   indicate what additional information would be needed.
5. For experimental design advice, provide step-by-step recommendations where possible.
6. Use SI units and standard notation appropriate to the field.
7. Answer in the same language as the question (Chinese or English).
""").strip()

_RAG_TEMPLATE = textwrap.dedent("""
RETRIEVED CONTEXT FROM KNOWLEDGE BASE:
{context}

---

USER QUESTION:
{question}

Please answer based primarily on the retrieved context above.
Cite specific sources (source file and section) when drawing on them.
If the context is insufficient, supplement with your general expertise and say so.
""").strip()

_NO_CONTEXT_TEMPLATE = textwrap.dedent("""
NOTE: No highly relevant passages were found in the knowledge base for this question.
The answer below is based on general expertise.

USER QUESTION:
{question}
""").strip()


class Assistant:
    """Interactive Q&A assistant backed by the knowledge base."""

    def __init__(
        self,
        kb: KnowledgeBase,
        domain: DomainContext,
        model: str = "gemini-2.5-flash",
        n_chunks: int = 6,
    ) -> None:
        self.kb = kb
        self.domain = domain
        self.model = model
        self.n_chunks = n_chunks
        self.llm: LLMBackend = get_llm(model)
        # OpenAI-style {role, content} message history.
        self._history: list[dict] = []
        self._system = _SYSTEM_PROMPT_TEMPLATE.format(
            role_description=domain.qa_role_description.strip()
            or f"You are an expert in {domain.expert_field}."
        )

    def ask(self, question: str, stream: bool = True) -> str:
        """Ask a question, optionally streaming. Returns the full answer."""
        context = self.kb.context_for_query(question, n_chunks=self.n_chunks)
        user_text = (
            _RAG_TEMPLATE.format(context=context, question=question)
            if context
            else _NO_CONTEXT_TEMPLATE.format(question=question)
        )
        self._history.append({"role": "user", "content": user_text})

        if stream:
            collected: list[str] = []
            for chunk in self.llm.stream_chat(
                self._history, system=self._system,
                temperature=0.2, max_tokens=8192,
            ):
                print(chunk, end="", flush=True)
                collected.append(chunk)
            print()
            answer = "".join(collected)
        else:
            answer = self.llm.chat(
                self._history, system=self._system,
                temperature=0.2, max_tokens=8192,
            )

        self._history.append({"role": "assistant", "content": answer})
        return answer

    def stream_chunks(self, question: str):
        """Yield text chunks — for use with Streamlit st.write_stream()."""
        context = self.kb.context_for_query(question, n_chunks=self.n_chunks)
        user_text = (
            _RAG_TEMPLATE.format(context=context, question=question)
            if context
            else _NO_CONTEXT_TEMPLATE.format(question=question)
        )
        self._history.append({"role": "user", "content": user_text})
        collected: list[str] = []
        for chunk in self.llm.stream_chat(
            self._history, system=self._system,
            temperature=0.2, max_tokens=8192,
        ):
            collected.append(chunk)
            yield chunk
        self._history.append({"role": "assistant", "content": "".join(collected)})

    def reset_history(self) -> None:
        self._history.clear()
