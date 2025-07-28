import re

# -----------------------------
# Clean Raw Text for Processing
# -----------------------------
def clean_text(text):
    """
    Cleans raw text by removing unwanted artifacts such as footnotes, section titles,
    special characters, and extra whitespace.

    Args:
        text (str): The raw input text to be cleaned.

    Returns:
        str: The cleaned and normalized text.
    """
    # remove footnotes like [1], [2], etc
    text = re.sub(r"\[\s*\d+\s*\]", "", text)

    # remove sections like 'References', 'External links', etc
    text = re.split(r"(?i)references|external links|see also|notes", text)[0]

    # Remove non-breaking spaces and other unicode artifacts
    text = text.replace('\xa0', ' ').replace('\u200b', '')

    text = re.sub(r"Chapter\s+\d+\s*:\s*", "", text)
    text = re.sub(r"\[\]", "", text)

    # Normalize multiple spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text