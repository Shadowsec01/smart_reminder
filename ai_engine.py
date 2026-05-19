# =============================================================================
#  ai_engine.py  —  AI Engine for SmartTask Pro
#  Author  : [Your Full Name] | Reg No: [Your Reg Number] | Dept: [Department]
#  Purpose : Houses all Artificial Intelligence logic used in the application.
#
#  AI MODES / TECHNIQUES USED IN THIS MODULE
#  ──────────────────────────────────────────
#  1. Rule-Based AI (Expert System)
#     A classic AI paradigm where human-authored IF-THEN rules drive decisions.
#     Used here for: category classification and reminder scheduling.
#     Advantage: Transparent, explainable, fast, no training data needed.
#
#  2. Natural Language Processing (NLP)
#     A branch of AI that lets computers understand human text/speech.
#     Used here for: extracting dates, times, and cleaning task titles from
#     free-form English input (e.g. "Submit lab report by Friday 5pm").
#     Library: dateparser — multilingual, handles relative expressions.
#
#  3. Heuristic Scoring Algorithm
#     A problem-solving approach that uses practical rules of thumb (heuristics)
#     to find good-enough solutions fast. Not perfect but efficient and logical.
#     Used here for: computing a numeric priority score (0-10) from urgency
#     keywords + deadline proximity, then mapping it to High/Medium/Low.
#
#  4. Pattern Matching / Entity Extraction (lightweight NER)
#     Named Entity Recognition (NER) is an NLP subtask that locates and
#     classifies named entities. We use regex + keyword rules to extract tags.
#     Used here for: auto-tagging tasks with relevant labels.
# =============================================================================

import re                          # Regular expressions — pattern matching
import dateparser                  # NLP date parsing (pip install dateparser)
from datetime import datetime, timedelta
from typing import Optional, Tuple


class AIEngine:
    """
    Central AI engine for SmartTask Pro.

    Design pattern: a single class bundles all AI methods so the Flask app
    only needs one import and one instantiation ('ai = AIEngine()').
    Each method represents a distinct AI technique — documented inline.
    """

    # ─── 1. RULE-BASED KNOWLEDGE BASE ────────────────────────────────────────
    # These dictionaries act as the "knowledge base" of our Expert System.
    # A domain expert (us) defined which words belong to which categories.
    # The AI uses these rules at runtime — no neural network, no training.

    CATEGORY_KEYWORDS: dict[str, list[str]] = {
        "Study": [
            "study", "assignment", "homework", "exam", "test", "quiz",
            "lecture", "course", "class", "read", "research", "thesis",
            "project", "submit", "report", "lab", "practical", "tutorial",
            "notes", "revision", "presentation", "essay", "dissertation",
        ],
        "Work": [
            "meeting", "email", "call", "presentation", "client", "office",
            "work", "job", "deliver", "proposal", "review", "invoice",
            "schedule", "conference", "interview", "apply", "send",
        ],
        "Health": [
            "exercise", "gym", "doctor", "medicine", "workout", "run",
            "jog", "diet", "health", "hospital", "appointment", "sleep",
            "rest", "yoga", "physio", "checkup", "drug", "prescription",
        ],
        "Finance": [
            "pay", "bill", "bank", "transfer", "budget", "money", "fee",
            "salary", "rent", "tuition", "subscription", "loan", "tax",
        ],
        "Tech": [
            "code", "program", "debug", "install", "update", "configure",
            "deploy", "build", "fix", "api", "database", "server",
            "script", "git", "push", "pull", "commit", "test",
        ],
        "Personal": [
            "buy", "shop", "visit", "family", "friend", "birthday",
            "clean", "cook", "travel", "plan", "book", "order", "pick",
            "party", "celebrate", "gift",
        ],
    }

    # Urgency keywords map to priority tiers.
    # Higher weight = stronger pull toward that priority level.
    URGENCY_WEIGHTS: dict[str, dict[str, float]] = {
        "high": {
            "urgent": 3.0, "asap": 3.0, "immediately": 3.0,
            "critical": 3.0, "emergency": 3.0, "must": 2.0,
            "deadline": 2.0, "required": 2.0, "overdue": 3.5,
            "important": 1.5, "priority": 1.5,
        },
        "low": {
            "someday": -2.0, "eventually": -2.0, "maybe": -1.5,
            "optional": -2.0, "later": -1.5, "when possible": -1.5,
            "no rush": -2.0, "sometime": -1.5,
        },
    }

    # Date-time pattern regex for NLP title cleaning.
    # After dateparser extracts the date, we strip the date phrase from the title
    # so "Submit report by Friday 5pm" becomes "Submit report".
    DATE_PATTERNS: list[str] = [
        r"\b(by|on|at|before|due|until|till)\s+\w+.*",   # "by Friday 5pm"
        r"\b(today|tomorrow|yesterday)\b.*",               # standalone time words
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b.*",
        r"\b\d{1,2}[:/]\d{2}\s*(am|pm)?\b.*",            # times like 3:00pm
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d+\b.*",
        r"\bin\s+\d+\s+(hour|day|week|minute)s?\b.*",    # "in 2 days"
        r"\bthis\s+(week|month|morning|afternoon|evening|night)\b.*",
        r"\bnext\s+(week|month|monday|tuesday|wednesday|thursday|friday)\b.*",
    ]

    # ─── 2. NLP: NATURAL LANGUAGE PARSING ────────────────────────────────────

    def parse_natural_language(self, text: str) -> dict:
        """
        AI TECHNIQUE: Natural Language Processing (NLP)

        What it does:
          Converts a raw human sentence into structured data by extracting:
            - The due date/time (if mentioned)
            - A cleaned task title (date phrase removed)

        How it works:
          1. dateparser.parse() scans the entire string for recognisable
             date/time expressions using multilingual grammar rules and
             locale-aware patterns.  It handles:
               • Absolute: "December 5", "2025-01-15 9am"
               • Relative: "tomorrow", "next Monday", "in 3 hours"
               • Colloquial: "end of day", "midnight", "noon"
          2. We then apply regex substitution to remove the date phrase from
             the original text, leaving a clean task title.

        Args:
            text (str): Raw user input, e.g. "submit lab report by tomorrow 5pm"

        Returns:
            dict with keys:
              'due_date'    → datetime | None
              'clean_title' → str (date expression removed)
        """

        # dateparser settings:
        #   PREFER_DATES_FROM='future'  → if "Monday" is ambiguous, pick next Monday
        #   RETURN_TIME_AS_PERIOD=False → always return a full datetime, not just date
        settings = {
            "PREFER_DATES_FROM": "future",
            "RETURN_TIME_AS_PERIOD": False,
            "PREFER_DAY_OF_MONTH": "first",
        }

        due_date: Optional[datetime] = dateparser.parse(text, settings=settings)

        # Sanity check: reject dates too far in the past (likely a false positive)
        if due_date and due_date < datetime.now() - timedelta(days=1):
            due_date = None

        # Clean the title by removing date/time phrases using regex rules
        clean_title = text
        for pattern in self.DATE_PATTERNS:
            clean_title = re.sub(pattern, "", clean_title, flags=re.IGNORECASE).strip()

        # Remove leftover punctuation/whitespace artifacts after stripping
        clean_title = re.sub(r"\s{2,}", " ", clean_title).strip(" ,;:-")

        # If cleaning wiped everything, fall back to original text
        if not clean_title:
            clean_title = text

        return {
            "due_date": due_date,
            "clean_title": clean_title,
        }

    # ─── 3. RULE-BASED AI: CATEGORY CLASSIFICATION ───────────────────────────

    def predict_category(self, text: str) -> str:
        """
        AI TECHNIQUE: Rule-Based Classification (Expert System)

        What it does:
          Assigns a task to one of 6 categories: Study, Work, Health,
          Finance, Tech, Personal — or defaults to 'General'.

        How it works (Bag-of-Words scoring):
          1. Tokenise the input text into lowercase words.
          2. For each category, count how many of its keywords appear in
             the token set (this is the 'bag-of-words' concept — we don't
             care about word order, only presence/frequency).
          3. The category with the highest match count wins.
          4. Tie → default to 'General'.

        This is a simple but effective Rule-Based AI because:
          • The rules (keyword lists) encode domain knowledge explicitly.
          • No statistical model or GPU needed.
          • 100% interpretable — we can always explain *why* a category was chosen.

        Args:
            text (str): Task title or description.

        Returns:
            str: Category name.
        """

        tokens = set(text.lower().split())   # tokenise to a set (fast lookup O(1))
        scores: dict[str, int] = {}

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            # Count how many category keywords appear in the token set
            # Also check for multi-word phrases with 'in' operator on full text
            score = sum(
                1 for kw in keywords
                if kw in tokens or kw in text.lower()
            )
            scores[category] = score

        # Find the winning category
        best_category = max(scores, key=scores.get)   # type: ignore[arg-type]
        best_score = scores[best_category]

        # Only assign a category if at least one keyword matched
        return best_category if best_score > 0 else "General"

    # ─── 4. HEURISTIC SCORING: PRIORITY PREDICTION ───────────────────────────

    def predict_priority(
            self, text: str, due_date: Optional[datetime]
    ) -> Tuple[str, float]:
        """
        AI TECHNIQUE: Heuristic Scoring Algorithm

        What it does:
          Computes a priority score (0.0 – 10.0) for a task and maps it to
          one of three labels: High / Medium / Low.

        How it works (two-signal heuristic):

          SIGNAL A — Keyword Urgency (up to ±6 points):
            Scan the text for urgency/relaxedness keywords.
            Each matched keyword adjusts the base score by its weight.
            High-urgency words add positive weight; low-urgency subtract.

          SIGNAL B — Deadline Proximity (up to +4 points):
            How close is the due date?
              ≤ 2 hours   → +4.0  (extremely urgent)
              ≤ 12 hours  → +3.0
              ≤ 24 hours  → +2.5
              ≤ 3 days    → +2.0
              ≤ 7 days    → +1.0
              > 7 days    → +0.0
            No due date  →  0.0 (neutral)

          FINAL SCORE = base(5.0) + Signal_A + Signal_B, clamped to [0, 10]

          LABEL MAPPING:
            score ≥ 7.0  → High
            score ≥ 4.0  → Medium
            score <  4.0  → Low

        Args:
            text (str): Task text.
            due_date (datetime | None): Parsed due date.

        Returns:
            Tuple[str, float]: (priority_label, priority_score)
        """

        base_score: float = 5.0   # neutral starting point
        text_lower = text.lower()

        # — Signal A: keyword urgency scan —
        for kw, weight in self.URGENCY_WEIGHTS["high"].items():
            if kw in text_lower:
                base_score += weight

        for kw, weight in self.URGENCY_WEIGHTS["low"].items():
            if kw in text_lower:
                base_score += weight   # weights are negative for 'low' keywords

        # — Signal B: deadline proximity —
        if due_date:
            delta_seconds = (due_date - datetime.now()).total_seconds()

            if delta_seconds <= 0:
                # Already overdue — maximum urgency boost
                base_score += 4.0
            elif delta_seconds <= 7_200:      # ≤ 2 hours
                base_score += 4.0
            elif delta_seconds <= 43_200:     # ≤ 12 hours
                base_score += 3.0
            elif delta_seconds <= 86_400:     # ≤ 24 hours
                base_score += 2.5
            elif delta_seconds <= 259_200:    # ≤ 3 days
                base_score += 2.0
            elif delta_seconds <= 604_800:    # ≤ 7 days
                base_score += 1.0
            # else: > 7 days → no proximity boost

        # Clamp score to valid range [0.0, 10.0]
        final_score = max(0.0, min(10.0, base_score))

        # Map numeric score to human-readable priority label
        if final_score >= 7.0:
            label = "High"
        elif final_score >= 4.0:
            label = "Medium"
        else:
            label = "Low"

        return label, round(final_score, 2)

    # ─── 5. RULE-BASED DECISION SYSTEM: SMART REMINDER ───────────────────────

    def suggest_reminder(
            self, due_date: Optional[datetime], priority: str
    ) -> Optional[datetime]:
        """
        AI TECHNIQUE: Rule-Based Decision System (Production Rules)

        What it does:
          Recommends the best time to send a reminder notification based on
          the task's deadline and priority level.

        How it works (decision table):
          Priority | Lead Time Before Deadline
          ─────────────────────────────────────
          High     | 2 hours before
          Medium   | 6 hours before (or day-before morning if > 1 day away)
          Low      | 24 hours before

          If the computed reminder time is already in the past,
          we fall back to "now + 5 minutes" so the user is notified immediately.

          If there is no due date, we set a default reminder 1 day from now
          as a general nudge to set a deadline.

        This is a deterministic, rule-driven system — the same inputs always
        produce the same output (unlike probabilistic ML models).

        Args:
            due_date (datetime | None): Parsed due date.
            priority (str): "High" | "Medium" | "Low"

        Returns:
            datetime | None: Suggested reminder datetime.
        """

        now = datetime.now()

        if not due_date:
            # No deadline? Set a gentle reminder for tomorrow to prompt the user
            return now + timedelta(days=1)

        # Lead times per priority level (as timedelta objects)
        lead_times: dict[str, timedelta] = {
            "High":   timedelta(hours=2),
            "Medium": timedelta(hours=6),
            "Low":    timedelta(hours=24),
        }

        lead = lead_times.get(priority, timedelta(hours=6))
        reminder_time = due_date - lead

        # If the suggested reminder is in the past, use an immediate fallback
        if reminder_time <= now:
            reminder_time = now + timedelta(minutes=5)

        return reminder_time

    # ─── 6. PATTERN MATCHING / ENTITY EXTRACTION: AUTO-TAGGING ──────────────

    def extract_tags(self, text: str) -> list[str]:
        """
        AI TECHNIQUE: Pattern Matching & Lightweight Entity Extraction

        What it does:
          Automatically extracts meaningful hashtag-style labels from task text.
          This is a simplified version of Named Entity Recognition (NER).

        How it works:
          1. Check for explicit #hashtags in the input.
          2. Scan the text against a curated tag-trigger dictionary.
          3. Return a deduplicated list of up to 5 tags.

        NER in production systems (spaCy, BERT-NER) uses deep learning to
        identify entities like PERSON, ORG, DATE, etc.  Our rule-based
        approach achieves similar practical results for this domain
        without any model weights or GPU.

        Args:
            text (str): Task text.

        Returns:
            list[str]: List of tag strings (e.g. ["#urgent", "#school"]).
        """

        tags: set[str] = set()
        text_lower = text.lower()

        # Step 1: Extract explicit user-written hashtags (e.g. "#urgent")
        explicit_tags = re.findall(r"#(\w+)", text)
        tags.update(f"#{t.lower()}" for t in explicit_tags)

        # Step 2: Rule-based tag triggers
        tag_rules: dict[str, list[str]] = {
            "#urgent":   ["urgent", "asap", "immediately", "critical", "emergency"],
            "#school":   ["assignment", "exam", "submission", "course", "class", "lab"],
            "#meeting":  ["meeting", "conference", "call", "zoom", "teams"],
            "#deadline": ["deadline", "due", "submit", "by", "before"],
            "#health":   ["doctor", "gym", "exercise", "appointment", "medicine"],
            "#finance":  ["pay", "bill", "bank", "fee", "tuition", "money"],
            "#personal": ["birthday", "family", "friend", "celebrate", "gift"],
            "#tech":     ["code", "debug", "deploy", "server", "git", "bug"],
            "#review":   ["review", "check", "proofread", "edit", "revise"],
        }

        for tag, triggers in tag_rules.items():
            if any(trigger in text_lower for trigger in triggers):
                tags.add(tag)

        # Limit to 5 tags to keep UI clean
        return sorted(tags)[:5]

    # ─── 7. INSIGHT GENERATION: PRODUCTIVITY ANALYSIS ────────────────────────

    def generate_insight(self, stats: dict) -> str:
        """
        AI TECHNIQUE: Rule-Based Natural Language Generation (NLG)

        What it does:
          Generates a human-readable productivity insight based on task stats.
          NLG is the inverse of NLP — instead of understanding text, we produce it.

        How it works:
          Simple template-selection system (a form of Rule-Based NLG):
          • Evaluate key metrics (completion rate, overdue count).
          • Select the most appropriate pre-written insight template.
          • Fill in the template with dynamic values.

        Advanced NLG systems (GPT-4, T5) generate text token-by-token using
        transformer models. Our rule-based approach is lighter and sufficient
        for structured numeric data like task statistics.

        Args:
            stats (dict): Dictionary from /api/stats endpoint.

        Returns:
            str: A personalised productivity message.
        """

        rate = stats.get("completion_rate", 0)
        overdue = stats.get("overdue", 0)
        active = stats.get("active", 0)
        total = stats.get("total", 0)

        # Select insight based on completion rate and overdue count
        if total == 0:
            return "🚀 Welcome! Add your first task to get started."

        if overdue > 3:
            return (
                f"⚠️ You have {overdue} overdue tasks. "
                "Focus on clearing them before adding new ones."
            )

        if rate >= 90:
            return f"🏆 Outstanding! {rate}% completion rate. You're crushing it!"

        if rate >= 70:
            return (
                f"💪 Great work! {rate}% done. "
                f"{active} tasks left — keep the momentum going."
            )

        if rate >= 50:
            return (
                f"📈 You're halfway there ({rate}%). "
                "Break your remaining tasks into smaller steps."
            )

        if rate < 30 and total > 5:
            return (
                f"🎯 {rate}% completion. Start with your highest-priority task — "
                "one win builds momentum for the rest."
            )

        return f"📋 {active} active tasks. Stay focused and tackle them one at a time."