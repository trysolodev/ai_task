"""Turn raw conversation turns into structured, typed memories.

Two backends produce the *same* ``ExtractedMemory`` shape:

* ``LLMExtractor`` — Claude with structured outputs (json_schema), the primary
  path when ``ANTHROPIC_API_KEY`` is set. Handles implicit facts, corrections,
  and opinion nuance far better than rules.
* ``RuleExtractor`` — deterministic regex/heuristic patterns. The fallback when
  there's no key or the LLM call fails. Lower recall, zero dependencies.

Both emit *canonical keys* (e.g. ``employment.company``, ``location.city``,
``pet.name``) so the store can detect contradictions and supersede stale facts
by ``(scope, key)``. This module only proposes memories; supersession lives in
``memory_store.py``.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger("memory.extraction")

VALID_TYPES = ("fact", "preference", "opinion", "event")

# Keys whose latest value supersedes the previous one. Events accumulate.
SUPERSEDING_TYPES = ("fact", "preference", "opinion")


@dataclass
class ExtractedMemory:
    type: str
    key: str
    value: str
    subject: Optional[str] = None
    confidence: float = 0.6
    correction: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "ExtractedMemory":
        self.type = self.type if self.type in VALID_TYPES else "fact"
        self.key = _norm_key(self.key)
        self.value = (self.value or "").strip()
        if self.subject:
            self.subject = self.subject.strip() or None
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        return self


class Extractor(Protocol):
    def extract(
        self, messages: Sequence[Tuple[str, str, Optional[str]]], known_facts: Sequence[Dict[str, str]]
    ) -> List[ExtractedMemory]: ...


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_KEY_RE = re.compile(r"[^a-z0-9_.]+")
# A proper-noun phrase: a capitalized word optionally followed by up to 3 more
# capitalized words (with small connectors). Captures "Stripe", "San Francisco",
# "Goldman Sachs" without swallowing trailing lowercase ("...as a backend eng").
_PROPER = r"[A-Z][\w&.\-]*(?:\s+(?:of |and |the )?[A-Z][\w&.\-]*){0,3}"
_STOP_TAIL = re.compile(
    r"\b(right now|currently|now|these days|so far|at the moment|today|this morning)\b\.?$",
    re.IGNORECASE,
)


def _norm_key(key: str) -> str:
    key = (key or "").strip().lower().replace(" ", "_").replace("-", "_")
    key = _KEY_RE.sub("", key)
    return key or "fact"


def _clean_value(value: str) -> str:
    value = value.strip().strip(".,;:!?\"'()")
    value = _STOP_TAIL.sub("", value).strip().strip(".,;:")
    return value.strip()


def _topic_key(prefix: str, topic: str) -> str:
    topic = re.sub(r"[^a-z0-9 ]+", "", topic.lower()).strip()
    words = topic.split()[:3]
    return _norm_key(f"{prefix}.{'_'.join(words)}") if words else prefix


def _messages_text(messages: Sequence[Tuple[str, str, Optional[str]]], roles=("user",)) -> str:
    parts = [c for (r, c, _n) in messages if r in roles and c]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Rule-based extractor
# --------------------------------------------------------------------------- #
class RuleExtractor:
    """Deterministic patterns for the most common personal facts.

    Intentionally conservative on precision for ambiguous cues (implicit pet
    names, bare opinions) via lower confidence so the recall layer can rank
    them appropriately.
    """

    _DIET_WORDS = ("vegetarian", "vegan", "pescatarian", "omnivore", "carnivore", "kosher", "halal")
    _PET_KINDS = ("dog", "cat", "puppy", "kitten", "parrot", "hamster", "rabbit", "fish", "bird", "snake")
    _NAME_BLOCKLIST = {
        "a", "an", "the", "so", "not", "really", "also", "still", "just", "now",
        "based", "from", "going", "working", "trying", "looking", "feeling",
    }

    def extract(self, messages, known_facts=()) -> List[ExtractedMemory]:
        out: List[ExtractedMemory] = []
        for role, content, _name in messages:
            if role != "user" or not content:
                continue
            correction = bool(re.search(r"\b(actually|sorry|i meant|correction|not\s+\w+,)\b", content, re.I))
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", content):
                out.extend(self._scan(sentence, correction))
        # Dedup by (type, key, value).
        seen: set = set()
        unique: List[ExtractedMemory] = []
        for m in out:
            m.normalized()
            if not m.value:
                continue
            sig = (m.type, m.key, m.value.lower())
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(m)
        return unique

    def _scan(self, s: str, correction: bool) -> List[ExtractedMemory]:
        found: List[ExtractedMemory] = []

        def add(type_, key, value, subject=None, conf=0.7, attrs=None):
            value = _clean_value(value)
            if value:
                found.append(
                    ExtractedMemory(type_, key, value, subject, conf, correction, attrs or {})
                )

        # Employment company (handles both word orders)
        m = re.search(r"\bI\s+work\s+(?:[^.,]*?\s+)?at\s+(" + _PROPER + r")", s)
        if m:
            add("fact", "employment.company", m.group(1), conf=0.85)
        m = re.search(
            r"\bI\s+(?:just\s+|recently\s+)?(?:joined|started\s+(?:working\s+)?(?:at\s+)?)\s*(" + _PROPER + r")",
            s,
        )
        if m:
            add("fact", "employment.company", m.group(1), conf=0.85, attrs={"event": "started"})
        m = re.search(r"\bI(?:'m| am)\s+now\s+(?:at|with)\s+(" + _PROPER + r")", s)
        if m:
            add("fact", "employment.company", m.group(1), conf=0.85)
        # Employment role: "as a/an <role>" anywhere in the sentence (allows
        # abbreviations like PM / CTO as well as lowercase titles)
        m = re.search(r"\bas\s+(?:a|an)\s+([A-Za-z][A-Za-z ]*?)(?:\s+at\b|[.,!?]|$)", s)
        if m and len(m.group(1).split()) <= 4:
            add("fact", "employment.role", m.group(1), conf=0.8)
        # "I'm a <role> at <Company>"
        m = re.search(r"\bI(?:'m| am)\s+(?:a|an)\s+([a-z][a-z /-]+?)\s+at\s+(" + _PROPER + r")", s)
        if m:
            add("fact", "employment.role", m.group(1), conf=0.8)
            add("fact", "employment.company", m.group(2), conf=0.85)

        # Location
        m = re.search(r"\bI\s+(?:just\s+|recently\s+)?moved\s+to\s+(" + _PROPER + r")(?:\s+from\s+(" + _PROPER + r"))?", s)
        if m:
            add("fact", "location.city", m.group(1), conf=0.85, attrs={"event": "moved"})
            if m.group(2):
                add("fact", "location.origin", m.group(2), conf=0.7)
        m = re.search(r"\bI\s+(?:live|am\s+living|am\s+based)\s+in\s+(" + _PROPER + r")", s)
        if m:
            add("fact", "location.city", m.group(1), conf=0.85)
        m = re.search(r"\bI(?:'m| am)\s+from\s+(" + _PROPER + r")", s)
        if m:
            add("fact", "location.origin", m.group(1), conf=0.7)

        # Pets (explicit)
        for kind in self._PET_KINDS:
            m = re.search(rf"\b(?:my|a)\s+{kind}\s+(?:named|called)\s+([A-Z]\w+)", s)
            if m:
                add("fact", "pet.name", m.group(1), subject=kind, conf=0.85)
                add("fact", "pet.type", kind, subject=m.group(1), conf=0.8)
            m = re.search(rf"\bI\s+have\s+a\s+{kind}\s+(?:named|called)\s+([A-Z]\w+)", s)
            if m:
                add("fact", "pet.name", m.group(1), subject=kind, conf=0.85)
                add("fact", "pet.type", kind, subject=m.group(1), conf=0.8)
        # Pets (implicit: "walking Biscuit") — verb case-insensitive, name must be Capitalized
        m = re.search(r"\b(?i:walking|walked|feeding|fed|petting|grooming)\s+([A-Z]\w+)\b", s)
        if m and m.group(1).lower() not in self._NAME_BLOCKLIST:
            add("fact", "pet.name", m.group(1), subject="pet", conf=0.5, attrs={"implicit": True})

        # Diet & allergies
        for d in self._DIET_WORDS:
            if re.search(rf"\bI(?:'m| am)\s+(?:a\s+)?{d}\b", s, re.I):
                add("preference", "diet", d, conf=0.85)
        m = re.search(r"\bI(?:'m| am)\s+allergic\s+to\s+([\w ,&]+)", s, re.I)
        if m:
            for allergen in re.split(r",|\band\b", m.group(1)):
                add("fact", "allergy", allergen, conf=0.85)
        m = re.search(r"\bI\s+(?:don't|do not|can't|cannot|avoid)\s+eat(?:ing)?\s+([\w ,&]+)", s, re.I)
        if m:
            for item in re.split(r",|\band\b", m.group(1)):
                add("preference", "diet.avoid", item, conf=0.7)

        # Family
        for rel, key in (("wife", "family.spouse"), ("husband", "family.spouse"),
                         ("partner", "family.partner"), ("spouse", "family.spouse")):
            m = re.search(rf"\bmy\s+{rel}\s+(?:is\s+)?([A-Z]\w+)", s)
            if m:
                add("fact", key, m.group(1), subject=rel, conf=0.8)
        for rel, key in (("son", "family.child"), ("daughter", "family.child")):
            m = re.search(rf"\bmy\s+{rel}\s+(?:is\s+)?([A-Z]\w+)", s)
            if m:
                add("fact", key, m.group(1), subject=rel, conf=0.8)

        # Name — case-insensitive prefix, but the captured name must be Capitalized
        # so "I am vegetarian" doesn't read as a name.
        m = re.search(r"\b(?i:my name is|i'?m|i am|call me)\s+([A-Z][a-z]{1,20})\b", s)
        if m and m.group(1).lower() not in self._NAME_BLOCKLIST and m.group(1).lower() not in self._DIET_WORDS:
            add("fact", "identity.name", m.group(1), conf=0.6)

        # Preferences
        if re.search(r"\b(concise|brief|short|to the point|direct)\b.*\b(answers?|responses?|replies)\b", s, re.I) or \
           re.search(r"\b(answers?|responses?|replies)\b.*\b(concise|brief|short|direct)\b", s, re.I):
            add("preference", "communication_style", "concise/direct", conf=0.75)
        m = re.search(r"\bI\s+prefer\s+([\w ,'+#.\-]+)", s, re.I)
        if m:
            add("preference", _topic_key("preference", m.group(1)), _clean_value(m.group(1)), conf=0.7)

        # Opinions
        m = re.search(r"\bI\s+(love|really like|like|enjoy|hate|dislike|can't stand)\s+([A-Za-z][\w +#.\-]*)", s, re.I)
        if m:
            stance = m.group(1).lower()
            topic = m.group(2)
            add("opinion", _topic_key("opinion", topic), f"{stance} {topic}".strip(), subject=topic, conf=0.65)
        m = re.search(r"\b([A-Z][\w +#.]{1,30})\s+(?:is|are)\s+(great|amazing|awful|terrible|annoying|overrated|underrated|the best|fine|frustrating)\b", s)
        if m:
            add("opinion", _topic_key("opinion", m.group(1)), f"{m.group(1)} is {m.group(2)}", subject=m.group(1), conf=0.6)

        # Goals / events
        m = re.search(r"\bI(?:'m| am)\s+(?:preparing|prepping|getting ready)\s+for\s+([\w ,'+#.\-]+)", s, re.I)
        if m:
            add("event", _topic_key("event", m.group(1)), f"preparing for {_clean_value(m.group(1))}", conf=0.6)

        return found


# --------------------------------------------------------------------------- #
# LLM extractor (Claude + structured outputs)
# --------------------------------------------------------------------------- #
_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(VALID_TYPES)},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "subject": {"type": "string"},
                    "confidence": {"type": "number"},
                    "correction": {"type": "boolean"},
                },
                "required": ["type", "key", "value", "subject", "confidence", "correction"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You extract durable, structured memories about a user from a conversation turn.

Return ONLY memories worth remembering across future conversations. Ignore small talk, \
acknowledgements, and transient task chatter.

For each memory choose:
- type: one of fact | preference | opinion | event
  * fact: stable personal facts (name, job, location, family, pets, diet, allergies, health)
  * preference: how the user likes things (communication style, dietary preference, tooling)
  * opinion: a subjective stance about a topic that may evolve over time
  * event: something that happened or is planned at a point in time (does not overwrite prior facts)
- key: a canonical, lowercase, dotted topic id. Reuse these where they fit:
  identity.name, employment.company, employment.role, employment.status, location.city,
  location.origin, education.school, family.spouse, family.partner, family.child, pet.name,
  pet.type, diet, allergy, health.condition, preference.communication_style,
  preference.<topic>, opinion.<topic>, skill.<topic>, goal.<topic>, event.<short>.
  Two memories about the SAME attribute MUST share the same key so newer ones supersede older.
- value: the concise current value (e.g. "Notion", "Berlin", "Biscuit", "vegetarian").
- subject: the entity the memory is about if not the user themselves (e.g. a pet's species
  "dog", a child's name); otherwise "".
- confidence: 0..1. Explicit statements ~0.9; reasonable inferences ~0.6; weak guesses ~0.4.
- correction: true if the user is correcting an earlier statement ("actually...", "I meant...").

Capture IMPLICIT facts too: "walking Biscuit this morning" -> pet.name = Biscuit (subject "dog" \
if known, else "pet"). "moved to Berlin from NYC" -> location.city = Berlin AND \
location.origin = NYC. Always return the user's CURRENT value for contradictions; the storage \
layer keeps the history.

Known active facts about this user (for key consistency and contradiction detection):
{known}

If nothing is worth remembering, return {{"memories": []}}."""


class LLMExtractor:
    def __init__(self, model: str, api_key: str, base_url: str = "") -> None:
        import anthropic

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model

    def extract(self, messages, known_facts=()) -> List[ExtractedMemory]:
        convo = self._format_turn(messages)
        if not convo.strip():
            return []
        known = "\n".join(f"- {f['key']}: {f['value']}" for f in known_facts) or "(none yet)"
        system = _SYSTEM_PROMPT.format(known=known)

        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": convo}],
            output_config={"format": {"type": "json_schema", "schema": _EXTRACTION_SCHEMA}},
        )
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        data = json.loads(text)
        out: List[ExtractedMemory] = []
        for item in data.get("memories", []):
            subject = (item.get("subject") or "").strip() or None
            out.append(
                ExtractedMemory(
                    type=item.get("type", "fact"),
                    key=item.get("key", ""),
                    value=item.get("value", ""),
                    subject=subject,
                    confidence=float(item.get("confidence", 0.6)),
                    correction=bool(item.get("correction", False)),
                ).normalized()
            )
        return [m for m in out if m.value]

    @staticmethod
    def _format_turn(messages) -> str:
        lines = []
        for role, content, name in messages:
            if not content:
                continue
            label = role.upper()
            if name:
                label += f" ({name})"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestrator: LLM with rule-based fallback
# --------------------------------------------------------------------------- #
class HybridExtractor:
    def __init__(self, llm: Optional[LLMExtractor], rules: RuleExtractor, backend: str) -> None:
        self._llm = llm
        self._rules = rules
        self._backend = backend

    @property
    def mode(self) -> str:
        if self._llm is not None:
            return "llm"
        return "rules"

    def extract(self, messages, known_facts=()) -> List[ExtractedMemory]:
        if self._llm is not None:
            try:
                return self._llm.extract(messages, known_facts)
            except Exception as exc:  # pragma: no cover - network/parse failures
                logger.warning("LLM extraction failed, falling back to rules: %s", exc)
        return self._rules.extract(messages, known_facts)


def build_extractor(settings) -> HybridExtractor:
    rules = RuleExtractor()
    backend = settings.extraction_backend
    use_llm = backend == "llm" or (backend == "auto" and bool(settings.anthropic_api_key))
    llm: Optional[LLMExtractor] = None
    if use_llm:
        if not settings.anthropic_api_key:
            logger.warning("EXTRACTION_BACKEND=llm but no ANTHROPIC_API_KEY; using rules")
        else:
            try:
                llm = LLMExtractor(
                    settings.extraction_model,
                    settings.anthropic_api_key,
                    base_url="",  # anthropic SDK reads ANTHROPIC_BASE_URL from env
                )
                logger.info("extraction backend: LLM (%s)", settings.extraction_model)
            except Exception as exc:  # pragma: no cover
                logger.warning("failed to init LLM extractor (%s); using rules", exc)
    if llm is None:
        logger.info("extraction backend: rule-based")
    return HybridExtractor(llm, rules, backend)
