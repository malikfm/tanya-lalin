import re


PAGE_PATTERN = re.compile(r"^-\s\d+\s-$")
CHAPTER_PATTERN = re.compile(r"^BAB\s[A-Z]+$")
SECTION_PATTERN = re.compile(r"^Bagian\s[A-Z][a-z]+$")
SUBSECTION_PATTERN = re.compile(r"^Paragraf\s\d+$")
ARTICLE_PATTERN = re.compile(r"^Pasal\s\d+$")
ARTICLE_WO_NUMBER_PATTERN = re.compile(r"^Pasal$")
BODY_PARAGRAPH_PATTERN = re.compile(r"^\(\d+\)$")
ELUCIDATION_PARAGRAPH_PATTERN = re.compile(r"^Ayat \(\d+\)$")
BODY_ORDERED_LIST_PATTERN = re.compile(r"^[a-z]\.$|^[a-z]\.\s[a-zA-Z]|^\d+\.$|^\d+\.\s[a-zA-Z]")
ELUCIDATION_ORDERED_LIST_PATTERN = re.compile(
    r"^[a-z]\.$|^[a-z]\.\s[a-zA-Z]|^\d+\.$|^\d+\.\s[a-zA-Z]|^Huruf\s[a-z]$|^Huruf\s[a-z]\s[a-zA-Z]"
)
TRIPPLE_DOTS_PATTERN = re.compile(r".+\.\.\.$")
TRIPPLE_DOT_SPACES_PATTERN = re.compile(r".+\.\s\.\s\.$")

END_OF_BODY_MARKER = "Disahkan di Jakarta"
END_OF_ELUCIDATION_MARKER = "TAMBAHAN LEMBARAN NEGARA REPUBLIK INDONESIA NOMOR 5025"
