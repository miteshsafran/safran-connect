from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Chunk configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Split loaded documents into smaller chunks.
    """

    chunks = []

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        split_texts = text_splitter.split_text(text)

        for index, chunk_text in enumerate(split_texts):

            chunk_metadata = metadata.copy()

            chunk_metadata["chunk_id"] = index

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                }
            )

    return chunks