# ResearchPilot AI

**An AI-powered assistant for understanding research papers.** Upload a PDF, ask questions in a chat box, and get answers grounded in the document, with the source pages and exact retrieved text shown for every answer.

🔗 **Live demo:** https://researchpilot-muskan.streamlit.app

> The demo runs on a free tier and may take a few seconds to wake up. It may also be rate-limited if many people use it at once.


## Why I built it
Research papers are long and dense, and general chatbots can invent details that aren't in the paper. ResearchPilot AI answers only from the uploaded document and shows its evidence, so every answer can be checked against the source.

## How it works
The app uses **Retrieval-Augmented Generation (RAG)**. No model is trained or fine-tuned. Both the embedding model and the language model are used as they are, so the tool works immediately on any new PDF.

1. **Extract:** PyPDF reads the text of the uploaded PDF page by page.
2. **Chunk:** LangChain text splitters break the text into chunks.
3. **Embed and index:** Sentence Transformers convert each chunk into a vector, and the vectors are stored in a FAISS index.
4. **Retrieve:** the question is converted to a vector, and the 5 most similar chunks are found.
5. **Generate:** the question and the retrieved chunks are sent to Gemini with strict instructions to answer only from the document.
6. **Stream and show evidence:** the answer streams to the screen, along with the pages searched and the exact text used.

## Features
- Answers grounded in the uploaded paper, which reduces made-up details
- Source pages and retrieved passages displayed with every answer
- **[Page selection: describe how a user limits the search to specific pages]**
- Streaming responses in a chat-style interface
- Uses **[three Gemini models: describe how, e.g. fallback if one fails, or one per task]**
- Error handling that shows clear messages instead of raw errors
- Works on any new PDF without preparation

## Tech stack
| Part | Tool |
|---|---|
| Interface | Streamlit |
| PDF parsing | PyPDF |
| Text splitting | LangChain text splitters |
| Embeddings | Sentence Transformers |
| Vector search | FAISS |
| Language model | Google Gemini (google-genai SDK) |
| Language | Python |

## Project structure
```
ResearchPilotAI/
├── app.py            # Streamlit app
├── utils/            # [describe: e.g. PDF loading, chunking, retrieval, Gemini calls]
├── requirements.txt
└── .gitignore
```

## Run locally
1. Clone the repo:
   `git clone https://github.com/Muskan-Sh-23/ResearchPilotAI.git`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Create a `.env` file in the project folder:
   `GEMINI_API_KEY=your_api_key`
4. Start the app:
   `streamlit run app.py`

Get a free API key from [Google AI Studio](https://aistudio.google.com).

## Limitations
- Works best on text-based PDFs; scanned PDFs would need OCR
- Answer quality depends on how the text is chunked and on the retrieved passages

## Future improvements
- **OCR support:** read scanned PDFs, which currently return little or no text
- **Multi-document questions:** upload several papers and ask questions across them
- **Better retrieval:** add a re-ranking step or hybrid (keyword + vector) search to improve which passages are selected
- **Evaluation:** build a small test set of questions and answers to measure retrieval and answer quality
- **Saved chat history:** keep a conversation per paper so users can return to it
- **Paper summaries:** one-click summary of the abstract, methods, and results
- **Persistent index:** save the FAISS index so a document doesn't have to be re-processed each session

## Author
**Muskan Sharma**, B.Tech AI & ML, UIET Kurukshetra University
[LinkedIn](https://www.linkedin.com/in/muskan-sharma26/) · muskansawarni@gmail.com
