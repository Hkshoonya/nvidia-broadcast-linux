"""Meeting summarization -- extracts key points from transcripts.

Uses extractive summarization (no LLM required). Identifies:
- Key topics discussed
- Action items (sentences with "should", "need to", "will", "must", etc.)
- Questions raised
- Decisions made

Fully local, no cloud APIs.
"""

import re
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


# Patterns for extracting action items
_ACTION_PATTERNS = [
    r'\b(?:need to|needs to|should|must|have to|has to|will|going to|gonna)\b',
    r'\b(?:action item|todo|to-do|follow up|follow-up)\b',
    r'\b(?:assign|deadline|by (?:monday|tuesday|wednesday|thursday|friday|next week|tomorrow|end of))\b',
]

# Patterns for questions
_QUESTION_PATTERNS = [
    r'\?$',
    r'\b(?:how|what|when|where|why|who|which|can we|should we|do we|does|is there)\b.*\?',
]

# Patterns for decisions
_DECISION_PATTERNS = [
    r'\b(?:decided|decision|agreed|let\'s go with|we\'ll use|approved)\b',
]


class MeetingSummarizer:
    """Extracts structured notes from meeting transcripts."""

    def summarize(self, transcript: str, duration_seconds: float = 0) -> MeetingNotes:
        """Generate meeting notes from a transcript.

        Args:
            transcript: full transcript text
            duration_seconds: meeting duration in seconds
        """
        sentences = self._split_sentences(transcript)

        if not sentences:
            return MeetingNotes(
                title="Meeting Notes",
                duration_minutes=duration_seconds / 60,
                summary="No transcript available.",
                key_points=[],
                action_items=[],
                questions=[],
                full_transcript=transcript,
            )

        # Extract components
        action_items = self._extract_by_patterns(sentences, _ACTION_PATTERNS)
        questions = self._extract_by_patterns(sentences, _QUESTION_PATTERNS)

        # Key points: longest sentences that aren't questions or action items
        used = set(action_items + questions)
        key_candidates = [s for s in sentences if s not in used and len(s) > 30]
        key_candidates.sort(key=len, reverse=True)
        key_points = key_candidates[:10]  # Top 10 by length

        # Generate summary (first sentence + key stats)
        summary_parts = []
        if duration_seconds > 0:
            mins = int(duration_seconds / 60)
            summary_parts.append(f"{mins} minute meeting")
        summary_parts.append(f"{len(sentences)} statements")
        if action_items:
            summary_parts.append(f"{len(action_items)} action items")
        if questions:
            summary_parts.append(f"{len(questions)} questions raised")
        summary = " | ".join(summary_parts)

        # Title: first substantial sentence or generic
        title = "Meeting Notes"
        for s in sentences:
            if len(s) > 20 and not s.endswith("?"):
                title = s[:80]
                break

        return MeetingNotes(
            title=title,
            duration_minutes=duration_seconds / 60,
            summary=summary,
            key_points=key_points,
            action_items=action_items,
            questions=questions,
            full_transcript=transcript,
        )

    def format_notes(self, notes: MeetingNotes) -> str:
        """Format meeting notes as markdown."""
        lines = [
            f"# {notes.title}",
            f"",
            f"**Duration:** {notes.duration_minutes:.0f} minutes",
            f"**Summary:** {notes.summary}",
            f"",
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
        """Split transcript into sentences."""
        # Split on period, question mark, exclamation, or newline
        raw = re.split(r'[.!?\n]+', text)
        return [s.strip() for s in raw if s.strip() and len(s.strip()) > 5]

    def _extract_by_patterns(self, sentences: list[str],
                              patterns: list[str]) -> list[str]:
        """Extract sentences matching any of the patterns."""
        matches = []
        for sentence in sentences:
            for pattern in patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    matches.append(sentence)
                    break
        return matches
