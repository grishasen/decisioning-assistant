from __future__ import annotations


def qa_generation_prompt(context: str, qa_per_chunk: int) -> str:
    return f"""
You are generating training data for an English instruction-tuned model.
Create exactly {qa_per_chunk} question-answer pairs grounded only in the provided context.
Do not invent facts.

Return strict JSON with this schema only:
{{
  "qa_pairs": [
    {{"question": "...", "answer": "..."}}
  ]
}}

Context:
{context}
""".strip()


def answer_with_context_prompt(question: str, context: str) -> str:
    return f"""
You are a precise assistant.
Use only the context below. If the context is insufficient, say you don't know.

Context:
{context}

Question:
{question}
""".strip()
