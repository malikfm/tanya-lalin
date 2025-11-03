import re


# Patterns for identifying document structural elements
PAGE_PATTERN = re.compile(r"^-\s\d+\s-$")  # Page numbers like "- 1 -"
CHAPTER_PATTERN = re.compile(r"^BAB\s[A-Z]+$")  # Chapters like "BAB I"
SECTION_PATTERN = re.compile(r"^Bagian\s[A-Z][a-z]+$")  # Sections like "Bagian Kesatu"
SUBSECTION_PATTERN = re.compile(r"^Paragraf\s\d+$")  # Subsections like "Paragraf 1"
ARTICLE_PATTERN = re.compile(r"^Pasal\s\d+$")  # Articles like "Pasal 1"
ARTICLE_WO_NUMBER_PATTERN = re.compile(r"^Pasal$")  # Article headers without number like "Pasal"

# Patterns for paragraph identification in different document sections
BODY_PARAGRAPH_PATTERN = re.compile(r"^\(\d+\)$")  # Body paragraphs like "(1)"
ELUCIDATION_PARAGRAPH_PATTERN = re.compile(r"^Ayat \(\d+\)$")  # Elucidation paragraphs like "Ayat (1)"

# Patterns for ordered list items in different document sections
BODY_ORDERED_LIST_PATTERN = re.compile(
    r"^[a-z]\.$|^[a-z]\.\s[a-zA-Z]|^\d+\.$|^\d+\.\s[a-zA-Z]"
)  # Body ordered lists like "a.", "b. text", "1.", "2. text"
ELUCIDATION_ORDERED_LIST_PATTERN = re.compile(
    r"^[a-z]\.$|^[a-z]\.\s[a-zA-Z]|^\d+\.$|^\d+\.\s[a-zA-Z]|^Huruf\s[a-z]$|^Huruf\s[a-z]\s[a-zA-Z]"
)  # Elucidation ordered lists including "Huruf a" patterns

# Patterns for skipping page delimiters
TRIPPLE_DOTS_PATTERN = re.compile(r".+\.\.\.$")  # Text ending with "...$"
TRIPPLE_DOT_SPACES_PATTERN = re.compile(r".+\.\s\.\s\.$")  # Text ending with ". . ."

# Markers for document section boundaries
END_OF_BODY_MARKER = "Disahkan di Jakarta"  # Marker indicating end of body content
END_OF_ELUCIDATION_MARKER = "TAMBAHAN LEMBARAN NEGARA REPUBLIK INDONESIA NOMOR 5025"  # Marker indicating end of elucidation content
