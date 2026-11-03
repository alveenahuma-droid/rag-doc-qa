# Document Q&A with RAG

A Retrieval-Augmented Generation (RAG) tool that answers questions about your PDF documents and cites the file and page behind each answer. It uses ChromaDB for retrieval and Claude to write the answer.

## Features

- Indexes any PDFs in the `docs` folder, page by page
- Retrieves the most relevant chunks for each question
- Answers using only the retrieved text, with citations such as `[handbook.pdf p.4]`
- Says "I couldn't find that in the documents." when the answer is not there
- Evaluation script that compares chunk sizes on answer accuracy and retrieval hit rate

## How It Works

1. **Ingest:** each PDF page is split into overlapping chunks and stored in ChromaDB, which creates the embeddings.
2. **Retrieve:** the question is embedded and the most similar chunks are returned with their source and page.
3. **Answer:** the chunks are given to Claude as context, with instructions to answer only from them and to cite sources.

## Requirements

- Python 3.9+
- An Anthropic API key
- PDFs with selectable text (not scanned images)

## Installation

```
git clone https://github.com/alveenahuma-droid/rag-doc-qa.git
cd rag-doc-qa
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
```

On Windows, use `setx ANTHROPIC_API_KEY your-key-here` and reopen the terminal.

## Usage

Put one or more PDFs in the `docs` folder, then:

```
python rag.py ingest
python rag.py ask "What is the late submission policy?"
```

The first ingest downloads a small embedding model, so it takes a minute.

## Evaluation

Edit `questions.json` with your own questions, the keywords a correct answer must contain, and the file and page of the answer. Then run:

```
python evaluate.py
```

It compares chunk sizes of 500 and 1000 characters and reports:

- **Answer accuracy:** all expected keywords appear in the answer
- **Retrieval hit rate:** the correct file and page were among the retrieved chunks

### Results

| Chunk size | Answer accuracy | Retrieval hit rate |
| --- | --- | --- |
| 500 | add result | add result |
| 1000 | add result | add result |

## Limitations

- Keyword matching is a simple measure of correctness
- Scanned PDFs are not supported
- Tables and images in PDFs are not handled

## Possible Improvements

- A web interface for uploading documents
- Re-ranking retrieved chunks
- Support for more file types

## License

MIT
