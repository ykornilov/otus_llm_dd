"""
BM25 tokenizer with Russian lemmatization via pymorphy2.

Russian words are reduced to their normal form (nominative singular etc.)
so that morphological variants match the same token:
  "корня" → "корень", "жирным" → "жирный", "округления" → "округление"

Latin words and function names (ROUND, LEFT, …) are lowercased but not lemmatized —
pymorphy2 does not handle Latin, and English function names are already invariant.
"""
import re

import pymorphy3

_morph = pymorphy3.MorphAnalyzer()
_RU_RE = re.compile(r"[а-яёА-ЯЁ]")
_TOK_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]+")


def tokenize(text: str) -> list[str]:
    """
    Split text into lowercase tokens; lemmatize Russian words.

    Examples:
        "квадратного корня"  → ["квадратный", "корень"]
        "жирным шрифтом"     → ["жирный", "шрифт"]
        "ROUND precision"    → ["round", "precision"]
    """
    tokens = []
    for word in _TOK_RE.findall(text.lower()):
        if _RU_RE.search(word):
            word = _morph.parse(word)[0].normal_form
        tokens.append(word)
    return tokens
