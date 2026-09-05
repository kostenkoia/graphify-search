"""BM25 over identifier-split tokens."""

from __future__ import annotations

import json
import math
from collections import Counter

# why: this tier answers without a model, so an index with no vectors still ranks

# why: the standard BM25 defaults from Robertson et al., k1 1.2 and b 0.75, which reference
# implementations ship
K1 = 1.2
B = 0.75


class BM25:
    """A BM25 index over pre-tokenised documents."""

    def __init__(self, ids: list[str], tf: list[dict[str, int]], df: dict[str, int]) -> None:
        self.ids = ids
        self._tf = tf
        self._df = df
        self._dl = [sum(c.values()) for c in tf]
        # inv: a zero avgdl is never divided by, because a document with a term has at least one token
        self._avgdl = (sum(self._dl) / len(self._dl)) if self._dl else 0.0
        n = len(ids)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    @classmethod
    def build(cls, docs: dict[str, list[str]]) -> BM25:
        """Build the index from `{doc id: tokens}`, keeping the dict's order."""
        ids = list(docs)
        tf = [dict(Counter(docs[i])) for i in ids]
        df: Counter[str] = Counter()
        for c in tf:
            df.update(c.keys())
        return cls(ids, tf, dict(df))

    def rank(self, query: list[str]) -> list[tuple[str, float]]:
        """Score every document against `query`; return non-zero scores, best first."""
        scores: dict[int, float] = {}
        for term in sorted(set(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, c in enumerate(self._tf):
                f = c.get(term)
                if not f:
                    continue
                norm = f + K1 * (1 - B + B * self._dl[i] / self._avgdl)
                scores[i] = scores.get(i, 0.0) + idf * f * (K1 + 1) / norm
        # inv: terms are summed in sorted order and ties fall to the smaller id, so two runs
        # print the same scores and order
        return sorted(((self.ids[i], s) for i, s in scores.items()), key=lambda x: (-x[1], x[0]))

    def to_json(self) -> str:
        """Serialise ids, term frequencies and document frequencies."""
        return json.dumps({"ids": self.ids, "tf": self._tf, "df": self._df}, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> BM25:
        """Rebuild an index written by `to_json`."""
        raw = json.loads(text)
        return cls(list(raw["ids"]), [dict(c) for c in raw["tf"]], dict(raw["df"]))
