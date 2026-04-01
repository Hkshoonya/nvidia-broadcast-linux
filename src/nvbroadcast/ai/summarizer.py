"""Meeting summarization -- extract structured notes from local transcripts.

This is intentionally heuristic and fully local. The main job is to turn a
rough transcript into usable meeting notes without promoting transcript noise,
greetings, or document boilerplate into fake action items.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class MeetingNotes:
    """Structured meeting notes."""

    title: str
    duration_minutes: float
    summary: str
    key_points: list[str]
    action_items: list[str]
    questions: list[str]
    full_transcript: str


_STRONG_ACTION_PATTERNS = [
    r"\b(?:we need to|need to|needs to|we should|should|let's|please|follow up|follow-up|todo|to-do|action item)\b",
    r"\b(?:assign|deadline|by (?:monday|tuesday|wednesday|thursday|friday|next week|tomorrow|end of))\b",
]

_QUESTION_PATTERNS = [
    r"\?$",
    r"\b(?:how|what|when|where|why|who|which|can we|should we|do we|does|is there)\b.*\?",
]

_DECISION_PATTERNS = [
    r"\b(?:decided|decision|agreed|let's go with|we'll use|approved|we chose)\b",
]

_FILLER_SENTENCES = {
    "hello",
    "hi",
    "okay",
    "ok",
    "yes",
    "yeah",
    "alright",
    "all right",
    "thank you",
    "thanks",
    "i'm sorry",
    "sorry",
}

_DOCUMENT_HINTS = {
    "form",
    "section",
    "signature",
    "address",
    "zip code",
    "certificate",
    "title number",
    "tax collector",
    "seller",
    "purchaser",
    "vehicle",
    "vessel",
    "mileage",
    "state law",
    "printed name",
    "date of sale",
}

_DOCUMENT_PRIORITY_HINTS = {
    "submit",
    "notice of sale",
    "bill of sale",
    "state law",
    "failure to complete",
    "mileage",
    "transfer of ownership",
    "under penalties of perjury",
    "declare",
}

_TERMS_HINTS = {
    "terms",
    "conditions",
    "trademark",
    "service provided",
    "not a bank",
    "card issuer",
    "rewards",
    "apple pay",
    "google pay",
    "samsung pay",
    "mastercard",
    "paypal",
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
    "you",
}


class MeetingSummarizer:
    """Extract structured notes from imperfect transcripts."""

    def summarize(self, transcript: str, duration_seconds: float = 0) -> MeetingNotes:
        sentences = self._split_sentences(transcript)
        cleaned_sentences = self._clean_sentences(sentences)

        if not cleaned_sentences:
            return MeetingNotes(
                title="Meeting Notes",
                duration_minutes=duration_seconds / 60,
                summary="No transcript available.",
                key_points=[],
                action_items=[],
                questions=[],
                full_transcript=transcript,
            )

        document_mode = self._looks_like_document_review(cleaned_sentences)
        terms_mode = not document_mode and self._looks_like_terms_review(cleaned_sentences)

        action_items = self._extract_action_items(cleaned_sentences, document_mode)
        questions = self._extract_by_patterns(cleaned_sentences, _QUESTION_PATTERNS)
        decisions = self._extract_by_patterns(cleaned_sentences, _DECISION_PATTERNS)

        used = set(action_items + questions + decisions)
        key_candidates = [s for s in cleaned_sentences if s not in used]
        if document_mode:
            key_points = self._rank_document_points(key_candidates, limit=8)
        elif terms_mode:
            key_points = self._rank_terms_points(key_candidates, limit=8)
        else:
            key_points = self._rank_key_points(key_candidates, limit=8)

        summary_parts = []
        if duration_seconds > 0:
            mins = max(1, int(duration_seconds / 60))
            summary_parts.append(f"{mins} minute meeting")
        summary_parts.append(f"{len(cleaned_sentences)} statements")
        if document_mode:
            summary_parts.append("document review")
        elif terms_mode:
            summary_parts.append("terms review")
        if action_items:
            summary_parts.append(f"{len(action_items)} action items")
        if questions:
            summary_parts.append(f"{len(questions)} questions raised")
        if decisions and not document_mode:
            summary_parts.append(f"{len(decisions)} decisions")
        summary = " | ".join(summary_parts)

        title = self._build_title(cleaned_sentences, document_mode, terms_mode)

        return MeetingNotes(
            title=title,
            duration_minutes=duration_seconds / 60,
            summary=summary,
            key_points=key_points,
            action_items=action_items,
            questions=questions[:6],
            full_transcript=transcript,
        )

    def format_notes(self, notes: MeetingNotes) -> str:
        lines = [
            f"# {notes.title}",
            "",
            f"**Duration:** {notes.duration_minutes:.0f} minutes",
            f"**Summary:** {notes.summary}",
            "",
        ]

        if notes.key_points:
            lines.append("## Key Points")
            lines.append("")
            for point in notes.key_points:
                lines.append(f"- {point}")
            lines.append("")

        if notes.action_items:
            lines.append("## Action Items")
            lines.append("")
            for item in notes.action_items:
                lines.append(f"- [ ] {item}")
            lines.append("")

        if notes.questions:
            lines.append("## Questions")
            lines.append("")
            for q in notes.questions:
                lines.append(f"- {q}")
            lines.append("")

        lines.append("## Full Transcript")
        lines.append("")
        lines.append(notes.full_transcript)
        return "\n".join(lines)

    def _split_sentences(self, text: str) -> list[str]:
        raw = re.split(r"[.!?\n]+", text)
        return [s.strip() for s in raw if s.strip() and len(s.strip()) > 2]

    def _clean_sentences(self, sentences: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen_norm: set[str] = set()
        for sentence in sentences:
            sentence = re.sub(r"\s+", " ", sentence).strip(" -,:;")
            if not sentence:
                continue
            if self._is_noise_sentence(sentence):
                continue
            if self._is_field_list_sentence(sentence):
                continue
            norm = self._normalize_sentence(sentence)
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            cleaned.append(sentence)
        return cleaned

    def _normalize_sentence(self, sentence: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", sentence.lower())).strip()

    def _is_noise_sentence(self, sentence: str) -> bool:
        norm = self._normalize_sentence(sentence)
        if not norm:
            return True
        if norm in _FILLER_SENTENCES:
            return True

        words = norm.split()
        if len(words) <= 2 and norm in _FILLER_SENTENCES:
            return True
        if len(words) >= 6:
            counts = Counter(words)
            top_count = counts.most_common(1)[0][1]
            unique_ratio = len(counts) / max(1, len(words))
            if top_count >= max(4, len(words) // 2):
                return True
            if unique_ratio < 0.35:
                return True
            repeated_bigram = self._has_repeated_phrase(words)
            if repeated_bigram:
                return True
        return False

    def _has_repeated_phrase(self, words: list[str]) -> bool:
        if len(words) < 8:
            return False
        phrases = Counter(zip(words, words[1:]))
        return phrases.most_common(1)[0][1] >= 4

    def _looks_like_document_review(self, sentences: list[str]) -> bool:
        doc_hits = 0
        for sentence in sentences:
            lower = sentence.lower()
            if any(hint in lower for hint in _DOCUMENT_HINTS):
                doc_hits += 1
        return len(sentences) >= 6 and (doc_hits / max(1, len(sentences))) >= 0.3

    def _looks_like_terms_review(self, sentences: list[str]) -> bool:
        term_hits = 0
        for sentence in sentences:
            lower = sentence.lower()
            if any(hint in lower for hint in _TERMS_HINTS):
                term_hits += 1
        return len(sentences) >= 5 and (term_hits / max(1, len(sentences))) >= 0.25

    def _is_field_list_sentence(self, sentence: str) -> bool:
        lower = sentence.lower()
        comma_count = sentence.count(",")
        if comma_count < 4:
            return False
        if re.search(r"\b(?:address|zip code|printed name|signature|date|city|state)\b", lower):
            return True
        words = re.findall(r"[a-z0-9]+", lower)
        if words:
            unique_ratio = len(set(words)) / max(1, len(words))
            if unique_ratio < 0.7 and comma_count >= 4:
                return True
        return False

    def _extract_by_patterns(self, sentences: list[str], patterns: list[str]) -> list[str]:
        matches: list[str] = []
        for sentence in sentences:
            for pattern in patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    matches.append(sentence)
                    break
        return matches[:8]

    def _extract_action_items(self, sentences: list[str], document_mode: bool) -> list[str]:
        matches: list[str] = []
        for sentence in sentences:
            lower = sentence.lower()
            if document_mode and any(hint in lower for hint in _DOCUMENT_HINTS):
                # Document instructions are usually not meeting tasks.
                if not re.search(r"\b(?:we need to|please|follow up|action item|todo|to-do|deadline)\b", lower):
                    continue
            if re.search(r"\b(?:i declare|state law requires|federal and state law)\b", lower):
                continue
            for pattern in _STRONG_ACTION_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    matches.append(sentence)
                    break
        return self._dedupe_semantic(matches)[:6]

    def _dedupe_semantic(self, sentences: list[str]) -> list[str]:
        result: list[str] = []
        seen: list[set[str]] = []
        for sentence in sentences:
            tokens = {
                token
                for token in re.findall(r"[a-z0-9]+", sentence.lower())
                if token not in _STOPWORDS and len(token) > 2
            }
            if not tokens:
                continue
            duplicate = False
            for existing in seen:
                overlap = len(tokens & existing) / max(1, len(tokens | existing))
                if overlap >= 0.75:
                    duplicate = True
                    break
            if duplicate:
                continue
            seen.append(tokens)
            result.append(sentence)
        return result

    def _rank_key_points(self, sentences: list[str], limit: int = 8) -> list[str]:
        if not sentences:
            return []

        token_counts: Counter[str] = Counter()
        sentence_tokens: list[tuple[str, list[str]]] = []
        for sentence in sentences:
            tokens = [
                token
                for token in re.findall(r"[a-z0-9]+", sentence.lower())
                if token not in _STOPWORDS and len(token) > 2
            ]
            sentence_tokens.append((sentence, tokens))
            token_counts.update(tokens)

        scored: list[tuple[float, str]] = []
        for sentence, tokens in sentence_tokens:
            if len(sentence) < 18:
                continue
            unique_tokens = set(tokens)
            if not unique_tokens:
                continue
            coverage = sum(math.log1p(token_counts[t]) for t in unique_tokens)
            length_bonus = min(len(sentence) / 140.0, 1.2)
            numeric_bonus = 0.2 if re.search(r"\d", sentence) else 0.0
            score = coverage + length_bonus + numeric_bonus
            scored.append((score, sentence))

        scored.sort(key=lambda item: item[0], reverse=True)
        ranked = self._dedupe_semantic([sentence for _, sentence in scored])
        return ranked[:limit]

    def _rank_document_points(self, sentences: list[str], limit: int = 8) -> list[str]:
        scored: list[tuple[float, str]] = []
        for sentence in sentences:
            lower = sentence.lower()
            if self._is_field_list_sentence(sentence):
                continue
            score = 0.0
            score += min(len(sentence) / 120.0, 1.0)
            score -= sentence.count(",") * 0.15
            if any(hint in lower for hint in _DOCUMENT_PRIORITY_HINTS):
                score += 1.4
            if re.search(r"\b(?:must|required|requires|submit|declare|certify)\b", lower):
                score += 0.9
            if re.search(r"\b(?:zip code|signature|printed name|city state)\b", lower):
                score -= 0.8
            if score > 0:
                scored.append((score, sentence))
        scored.sort(key=lambda item: item[0], reverse=True)
        ranked = self._dedupe_semantic([sentence for _, sentence in scored])
        return ranked[:limit]

    def _rank_terms_points(self, sentences: list[str], limit: int = 8) -> list[str]:
        scored: list[tuple[float, str]] = []
        for sentence in sentences:
            lower = sentence.lower()
            score = 0.0
            score += min(len(sentence) / 110.0, 1.0)
            if any(hint in lower for hint in _TERMS_HINTS):
                score += 0.9
            if re.search(r"\b(?:not a bank|issued by|issuer|linked to|redeemed|accepted|compatible devices)\b", lower):
                score += 0.8
            if re.search(r"\b(?:paypal|apple pay|google pay|samsung pay|mastercard)\b", lower):
                score += 0.7
            if re.search(r"\b(?:www|http|\.com|slash)\b", lower):
                score -= 0.6
            if sentence.count(",") >= 4:
                score -= 0.4
            if score > 0.4:
                scored.append((score, sentence))
        scored.sort(key=lambda item: item[0], reverse=True)
        ranked = self._dedupe_semantic([sentence for _, sentence in scored])
        return ranked[:limit]

    def _build_title(self, sentences: list[str], document_mode: bool, terms_mode: bool) -> str:
        if document_mode:
            title = self._build_document_title(sentences)
            if title:
                return title
            top = sentences[0]
            top = re.sub(r"\s+", " ", top).strip()
            return self._truncate_title(top)

        if terms_mode:
            title = self._build_terms_title(sentences)
            if title:
                return title

        for sentence in self._rank_key_points(sentences, limit=3):
            if len(sentence) >= 18:
                return self._truncate_title(sentence)
        return "Meeting Notes"

    def _build_document_title(self, sentences: list[str]) -> str | None:
        text = " ".join(sentences).lower()
        if "notice of sale" in text and "bill of sale" in text:
            return "Notice of sale and bill of sale form review"
        if "notice of sale" in text:
            return "Notice of sale form review"
        if "bill of sale" in text:
            return "Bill of sale form review"
        return None

    def _build_terms_title(self, sentences: list[str]) -> str | None:
        text = " ".join(sentences)
        labels: list[str] = []
        patterns = [
            (r"\bpaypal\b", "PayPal"),
            (r"\bapple pay\b", "Apple Pay"),
            (r"\bgoogle pay\b", "Google Pay"),
            (r"\bsamsung pay\b", "Samsung Pay"),
            (r"\bmastercard\b", "Mastercard"),
        ]
        for pattern, label in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                labels.append(label)
            if len(labels) == 2:
                break
        if labels:
            joined = " and ".join(labels) if len(labels) == 2 else labels[0]
            return f"{joined} terms overview"
        return None

    def _truncate_title(self, text: str, max_len: int = 76) -> str:
        text = text.strip()
        if len(text) <= max_len:
            return text
        trimmed = text[:max_len].rsplit(" ", 1)[0].strip()
        return trimmed or text[:max_len]
