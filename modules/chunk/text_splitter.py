import re


def split_text(text: str, max_length: int = 256, overlap: int = 50, min_length: int = None) -> list[str]:
    """
    Split text into chunks of at most `max_length` words, with `overlap` words overlapping between chunks.
    - First splits by h3 headings to respect natural section boundaries.
    - Then splits into sentences at punctuation boundaries (., ?, !).
    - Aggregates sentences into word-based chunks, handling sentences longer than max_length.
    """
    if not text or not text.strip():
        return []

    # Set default minimum length if not provided
    if min_length is None:
        min_length = max_length // 2

    initial_chunks: list[str] = []
    # 1. Split on h3 headings to respect natural section breaks
    segments = re.split(r'(?=\nh3\s+)', text)

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # 2. Split segment into sentences by punctuation
        sentences = re.split(r'(?<=[\.\?\!])\s+', seg)
        current_chunk: list[str] = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            words = sent.split()

            # If a single sentence exceeds max_length, flush current and slice sentence
            if len(words) > max_length:
                if current_chunk:
                    initial_chunks.append(" ".join(current_chunk))
                    prev = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk[:]
                    current_chunk = prev.copy()
                for i in range(0, len(words), max_length):
                    initial_chunks.append(" ".join(words[i : i + max_length]))
                current_chunk = []
                continue

            # If adding this sentence exceeds max_length, flush with overlap
            if len(current_chunk) + len(words) > max_length:
                initial_chunks.append(" ".join(current_chunk))
                prev = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk[:]
                current_chunk = prev.copy()

            current_chunk.extend(words)

        # Flush remaining words in current_chunk
        if current_chunk:
            initial_chunks.append(" ".join(current_chunk))

    # 3. Post-process: merge chunks smaller than min_length with the next chunk
    merged_chunks: list[str] = []
    i = 0
    while i < len(initial_chunks):
        current_words = initial_chunks[i].split()
        if len(current_words) < min_length and i + 1 < len(initial_chunks):
            next_words = initial_chunks[i + 1].split()
            merged = current_words + next_words
            # If merged chunk still too large, split again
            if len(merged) > max_length:
                merged_chunks.append(" ".join(merged[:max_length]))
                remaining = merged[max_length:]
                # Keep overlap for following chunks
                overlap_slice = merged[max_length - overlap : max_length]
                # Prepend overlap to remaining words
                initial_chunks[i + 1] = " ".join(overlap_slice + remaining)
            else:
                merged_chunks.append(" ".join(merged))
                i += 1  # Skip the next chunk since it's merged
        else:
            merged_chunks.append(initial_chunks[i])
        i += 1

    return merged_chunks
