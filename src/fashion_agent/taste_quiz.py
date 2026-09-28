"""Pairwise outfit feedback, kept separate from explicit Style DNA memories."""

import json
import os
import random
import re
import sqlite3
import uuid
from collections import Counter
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_card_image(image_path: str) -> Path:
    relative = Path(image_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Image paths must be relative paths inside outfit_cards")
    resolved = (PROJECT_ROOT / relative).resolve()
    if not resolved.is_relative_to((PROJECT_ROOT / "outfit_cards").resolve()):
        raise ValueError("Images must be inside outfit_cards")
    return resolved


class OutfitCard(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    image_url: str | None = Field(default=None, pattern=r"^https://[^\s]+$")
    image_path: str | None = None
    description: str = Field(min_length=1)
    attributes: list[str] = Field(min_length=1)
    duplicate_of: str | None = None
    annotation_source: str = "manual"

    @model_validator(mode="after")
    def validate_image(self):
        if bool(self.image_path) == bool(self.image_url):
            raise ValueError("Provide exactly one of image_path and image_url")
        if self.image_path:
            resolve_card_image(self.image_path)
        return self

    @property
    def public_image_url(self) -> str:
        if self.image_path:
            return f"/api/taste/images/{self.id}"
        return self.image_url or ""

    @field_validator("attributes")
    @classmethod
    def validate_attributes(cls, values: list[str]) -> list[str]:
        pattern = (
            r"^(color|silhouette|fit|material|pattern|detail|item|style):[a-z0-9_]+$"
        )
        if any(not re.fullmatch(pattern, value) for value in values):
            raise ValueError("Use normalized category:target attributes")
        return sorted(set(values))


def load_cards(path: Path | None = None) -> list[OutfitCard]:
    path = path or Path(
        os.environ.get(
            "CHERRY_TASTE_CARDS", str(PROJECT_ROOT / "data/taste_cards.json")
        )
    )
    if not path.exists():
        return []
    cards = [
        OutfitCard.model_validate(value)
        for value in json.loads(path.read_text(encoding="utf-8"))
    ]
    if len({card.id for card in cards}) != len(cards):
        raise ValueError("Card IDs must be unique")
    by_id = {card.id: card for card in cards}
    for card in cards:
        if card.image_path and not resolve_card_image(card.image_path).is_file():
            raise ValueError(f"Image file not found: {card.image_path}")
        if card.duplicate_of and (
            card.duplicate_of not in by_id
            or card.duplicate_of == card.id
            or by_id[card.duplicate_of].duplicate_of
        ):
            raise ValueError(f"Invalid duplicate_of for {card.id}")
    return cards


def inferred_preferences(votes: list[dict]) -> list[dict]:
    """A conservative heuristic, not a calibrated probability of liking an item.

    Only contrasts carry evidence. Shared attributes cannot explain the choice.
    Splitting a vote across its contrasts limits credit from complex outfits.
    """
    balance: Counter = Counter()
    evidence: Counter = Counter()
    for vote in votes:
        if vote["choice"] == "skip":
            continue
        winner, loser = (set(vote["left"]), set(vote["right"]))
        if vote["choice"] == "right":
            winner, loser = loser, winner
        contrast = winner ^ loser
        for attribute in contrast:
            weight = 1 / len(contrast)
            evidence[attribute] += weight
            balance[attribute] += weight if attribute in winner else -weight

    preferences = []
    for attribute, count in sorted(evidence.items()):
        if abs(balance[attribute]) < 1e-9:
            continue
        category, target = attribute.split(":", 1)
        preferences.append(
            {
                "category": category,
                "target": target,
                "polarity": "like" if balance[attribute] > 0 else "dislike",
                "strength": "weak",
                "confidence": round(abs(balance[attribute]) / (count + 2), 4),
                "source": "pairwise",
                "evidence_weight": round(count, 4),
            }
        )
    return preferences


class TasteQuiz:
    """SQLite stores issued pairs and immutable card snapshots for each answer."""

    def __init__(self, path: Path | None = None):
        self.path = path or Path(
            os.environ.get("CHERRY_TASTE_DB", "data/taste.sqlite3")
        )

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            db.execute("""CREATE TABLE IF NOT EXISTS taste_rounds (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, cards TEXT NOT NULL,
                choice TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS taste_dialogues (
                user_id TEXT PRIMARY KEY, data TEXT NOT NULL
            )""")
            with db:
                yield db
        finally:
            db.close()

    def preferences(self, user_id: str) -> list[dict]:
        return inferred_preferences(self.votes(user_id))

    def votes(self, user_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT cards, choice FROM taste_rounds WHERE user_id = ? AND choice IS NOT NULL ORDER BY rowid",
                (user_id,),
            ).fetchall()
        return [
                {
                    "choice": row["choice"],
                    "left": json.loads(row["cards"])[0]["attributes"],
                    "right": json.loads(row["cards"])[1]["attributes"],
                }
                for row in rows
            ]

    def dialogue(self, user_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT data FROM taste_dialogues WHERE user_id = ?", (user_id,)).fetchone()
        return json.loads(row["data"]) if row else {}

    def save_dialogue(self, user_id: str, state: dict) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO taste_dialogues (user_id, data) VALUES (?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET data=excluded.data",
                (user_id, json.dumps(state, ensure_ascii=False)),
            )

    def next_pair(self, user_id: str, cards: list[OutfitCard]) -> dict:
        cards = [card for card in cards if card.duplicate_of is None]
        with self._connect() as db:
            # Serializes simultaneous requests, so refreshes reuse the same round.
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT * FROM taste_rounds WHERE user_id = ? ORDER BY rowid",
                (user_id,),
            ).fetchall()
            answered = sum(row["choice"] is not None for row in rows)
            for row in rows:
                if row["choice"] is None:
                    return self._pair_payload(
                        row["id"], json.loads(row["cards"]), answered
                    )

            seen = set()
            exposure: Counter = Counter()
            for row in rows:
                previous = json.loads(row["cards"])
                ids = tuple(sorted(card["id"] for card in previous))
                seen.add(ids)
                exposure.update(ids)
            candidates = [
                (left, right)
                for left, right in combinations(cards, 2)
                if tuple(sorted((left.id, right.id))) not in seen
                and set(left.attributes) != set(right.attributes)
            ]
            if not candidates:
                return {"round_id": None, "cards": [], "answered": answered}
            # Explore the collection first; random ties and sides reduce position bias.
            random.shuffle(candidates)
            left, right = min(
                candidates, key=lambda pair: exposure[pair[0].id] + exposure[pair[1].id]
            )
            pair = [left.model_dump(), right.model_dump()]
            random.shuffle(pair)
            round_id = str(uuid.uuid4())
            db.execute(
                "INSERT INTO taste_rounds (id, user_id, cards) VALUES (?, ?, ?)",
                (round_id, user_id, json.dumps(pair)),
            )
            return self._pair_payload(round_id, pair, answered)

    @staticmethod
    def _pair_payload(round_id: str, cards: list[dict], answered: int) -> dict:
        return {
            "round_id": round_id,
            "cards": [
                {
                    "id": card["id"],
                    "description": card["description"],
                    "image_url": OutfitCard.model_validate(card).public_image_url,
                }
                for card in cards
            ],
            "answered": answered,
        }

    def answer(self, user_id: str, round_id: str, choice: str) -> None:
        if choice not in {"left", "right", "skip"}:
            raise ValueError("Выберите один образ или пропустите пару.")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT choice FROM taste_rounds WHERE id = ? AND user_id = ?",
                (round_id, user_id),
            ).fetchone()
            if row is None:
                raise ValueError("Эта пара не найдена для пользователя.")
            if row["choice"] is not None:
                if row["choice"] == choice:
                    return  # A retry must not count as additional evidence.
                raise ValueError("Ответ на эту пару уже сохранён.")
            db.execute(
                "UPDATE taste_rounds SET choice = ? WHERE id = ?", (choice, round_id)
            )
