import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash"]

SYSTEM_INSTRUCTION = """You are ResearchPilot, an assistant that helps students and researchers understand academic papers.

Rules:
1. Use ONLY the document context provided. Never use outside knowledge to fill gaps.
2. If the context does not contain the answer, reply exactly:
   "I could not find the answer in the provided document."
   If the context only partly answers the question, answer that part and say what is missing.
3. Cite the page for each key claim like (p. 4), using the [Page N] labels in the context.
4. Do not invent numbers, results, citations, or author names.
5. Explain clearly for a student: start with a direct answer, then add technical detail.
6. Use short paragraphs or bullet points for multi-part answers."""


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file or Streamlit secrets.")
    return genai.Client(api_key=api_key)


def generate_answer(question, context, retries_per_model=2):
    client = _get_client()

    prompt = f"""Document context:
{context}


Question:
{question}"""

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
    )

    last_error = None
    for model_name in MODELS:
        for attempt in range(retries_per_model + 1):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=prompt, config=config
                )
                return response.text or "I could not find the answer in the provided document."
            except Exception as e:
                last_error = e
                code = getattr(e, "code", None)
                if code in (429, 500, 503):
                    if attempt < retries_per_model:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    break  # this model is busy, try the next one
                raise  # other errors won't be fixed by retrying

    raise last_error

def generate_answer_stream(question, context, retries_per_model=1):
    client = _get_client()

    prompt = f"""Document context:
{context}

Question:
{question}"""

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
    )

    last_error = None
    for model_name in MODELS:
        for attempt in range(retries_per_model + 1):
            started = False
            try:
                stream = client.models.generate_content_stream(
                    model=model_name, contents=prompt, config=config
                )
                for chunk in stream:
                    if chunk.text:
                        started = True
                        yield chunk.text
                return
            except Exception as e:
                last_error = e
                if started:
                    raise  # failed mid-answer, don't restart and duplicate text
                code = getattr(e, "code", None)
                if code in (429, 500, 503):
                    if attempt < retries_per_model:
                        time.sleep(1)
                        continue
                    break
                raise

    raise last_error