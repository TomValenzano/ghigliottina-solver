"""Caricamento e ispezione del dataset della Ghigliottina (formato JSONL).

Ogni riga è un oggetto JSON con:
  - id, oldId
  - hint1..hint5 : i cinque indizi
  - sol          : la parola-soluzione
  - desc         : (opzionale) descrizione testuale della soluzione
  - ttg          : (opzionale) True se il gioco proviene dal gioco da tavolo
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Game:
    id: int
    hints: list[str]
    solution: str
    description: str | None = None
    is_board_game: bool = False
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def has_description(self) -> bool:
        return bool(self.description and self.description.strip())


def _parse_record(rec: dict) -> Game:
    hints = [rec.get(f"hint{i}", "") for i in range(1, 6)]
    return Game(
        id=rec.get("id", -1),
        hints=[h for h in hints if h],
        solution=rec.get("sol", "").strip(),
        description=(rec.get("desc") or None),
        is_board_game=bool(rec.get("ttg", False)),
        raw=rec,
    )


def load_games(path: str | Path) -> list[Game]:
    """Carica un file JSONL (un oggetto JSON per riga) o un array JSON."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    games: list[Game] = []
    stripped = text.lstrip()
    if stripped.startswith("["):  # array JSON
        for rec in json.loads(text):
            games.append(_parse_record(rec))
    else:  # JSONL
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            games.append(_parse_record(json.loads(line)))
    return games


def iter_games(path: str | Path) -> Iterator[Game]:
    yield from load_games(path)


if __name__ == "__main__":
    import sys

    g = load_games(sys.argv[1])
    print(f"{len(g)} partite caricate")
    print("Esempio:", g[0])
