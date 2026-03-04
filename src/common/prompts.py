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


def webex_thread_question_prompt(thread_start: str, replies: str) -> str:
    return f"""
You are generating English training data for an instruction-tuned technical assistant.
A Webex thread contains an initial message and reply messages.
Generate exactly one natural user question that is answered by the reply messages.
Use the whole thread for understanding, but do not answer the question.
Do not mention Webex, timestamps, participants, or that this came from a thread.
Do not invent facts outside the thread.

Return strict JSON with this schema only:
{{
  "question": "..."
}}

Thread start:
{thread_start}

Reply messages:
{replies}
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
