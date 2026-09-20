"""
Stage 2 of the pipeline: splitting documents into chunks.

⚠️ THIS IS THE FILE YOU CHANGE IN MILESTONE 3.

`split_documents` below is deliberately plain. It cuts every document into
fixed-size pieces with a fixed overlap and pays no attention to where sentences
or paragraphs end. It works, and it is not good.

On a corpus of short posts it may not cut anything at all: `campus_life` comes
out as 88 documents and 88 chunks, because almost nothing in it reaches 800
characters. That is the baseline, not a bug — Milestone 3 is where you decide
whether one post should stay one chunk.

Your job in Milestone 3 is to replace the *body* of `split_documents` with a
strategy that fits the documents you actually read in Milestone 1. Keep the
name and the shape of what it returns — the rest of the pipeline calls it, and
your README has to name the function that produced your chunks.

If you get stuck for 30 minutes, `fallback_split` is the original. Switch back
to it, write down what you saw, and move on. That's a real observation about
your pipeline, not giving up.
"""

from dataclasses import dataclass

import config
from ingest import Document
import re
import textwrap

@dataclass
class Chunk:
    """One piece of one document."""

    text: str
    source: str        # which file it came from
    index: int         # which chunk within that file, starting at 0
    produced_by: str   # the function that made it — cite this in your README

    @property
    def label(self) -> str:
        return f"{self.source}#{self.index}"


def fallback_split(
    documents: list[Document],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    The starter's original chunker. Fixed-size character windows with overlap.

    Keep this function. Milestone 3's stop rule points back at it, and having
    something to compare your own strategy against is useful in unit 2.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if overlap >= chunk_size:
        raise ValueError("overlap has to be smaller than chunk_size")

    chunks: list[Chunk] = []
    for doc in documents:
        start = 0
        index = 0
        while start < len(doc.text):
            piece = doc.text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::fallback_split",
                    )
                )
                index += 1
            start += chunk_size - overlap

    return chunks


def split_documents(documents: list[Document]) -> list[Chunk]:
    """
    Keep short advice threads together.

    Longer documents are split at paragraph and sentence boundaries so that
    chunks contain complete thoughts instead of arbitrary character windows.
    """

    max_chars = 800
    min_chunk_chars = 100
    produced_by = "chunker.py::split_documents"

    chunks: list[Chunk] = []

    for document in documents:
        text = document.text.strip()

        if not text:
            continue

        # Advice threads shorter than the limit stay as one complete chunk.
        if len(text) <= max_chars:
            chunks.append(
                Chunk(
                    text=text,
                    source=document.source,
                    index=0,
                    produced_by=produced_by,
                )
            )
            continue

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
        ]

        pieces: list[str] = []

        for paragraph in paragraphs:
            if len(paragraph) <= max_chars:
                pieces.append(paragraph)
                continue

            # Split unusually long paragraphs at sentence boundaries.
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            current_sentence_group = ""

            for sentence in sentences:
                sentence = sentence.strip()

                if not sentence:
                    continue

                # Handle a sentence longer than max_chars.
                if len(sentence) > max_chars:
                    if current_sentence_group:
                        pieces.append(current_sentence_group)
                        current_sentence_group = ""

                    pieces.extend(
                        textwrap.wrap(
                            sentence,
                            width=max_chars,
                            break_long_words=True,
                            break_on_hyphens=False,
                        )
                    )
                    continue

                candidate = (
                    f"{current_sentence_group} {sentence}".strip()
                    if current_sentence_group
                    else sentence
                )

                if (
                    current_sentence_group
                    and len(candidate) > max_chars
                ):
                    pieces.append(current_sentence_group)
                    current_sentence_group = sentence
                else:
                    current_sentence_group = candidate

            if current_sentence_group:
                pieces.append(current_sentence_group)

        # Combine paragraphs and sentences into chunks.
        document_chunks: list[str] = []
        current_chunk = ""

        for piece in pieces:
            candidate = (
                f"{current_chunk}\n\n{piece}".strip()
                if current_chunk
                else piece
            )

            if current_chunk and len(candidate) > max_chars:
                document_chunks.append(current_chunk)
                current_chunk = piece
            else:
                current_chunk = candidate

        if current_chunk:
            document_chunks.append(current_chunk)

        # Avoid a tiny final chunk when it can fit with the previous chunk.
        if len(document_chunks) >= 2:
            previous = document_chunks[-2]
            final = document_chunks[-1]
            merged = f"{previous}\n\n{final}"

            if len(final) < min_chunk_chars and len(merged) <= max_chars:
                document_chunks[-2] = merged
                document_chunks.pop()

        for index, chunk_text in enumerate(document_chunks):
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=document.source,
                    index=index,
                    produced_by=produced_by,
                )
            )

    return chunks

def describe(chunks: list[Chunk]) -> str:
    """A one-line summary, printed after indexing."""
    if not chunks:
        return "0 chunks"
    lengths = [len(c.text) for c in chunks]
    return (
        f"{len(chunks)} chunks, "
        f"{sum(lengths) // len(lengths)} characters on average "
        f"(shortest {min(lengths)}, longest {max(lengths)}), "
        f"produced by {chunks[0].produced_by}"
    )


if __name__ == "__main__":
    from ingest import load_documents

    chunks = split_documents(load_documents())
    print(describe(chunks))
