from __future__ import annotations

from datetime import datetime
import re


def build_cover_page(topic: str, author: str = "Autonomous Research Agent") -> str:
    """Generate a markdown cover page for the research report."""
    generated_on = datetime.now().strftime("%d %B %Y, %I:%M %p")
    cover = (
        "# Cover Page\n"
        "\n"
        f"**Report Title:** {topic}  \n"
        f"**Prepared By:** {author}  \n"
        f"**Generated On:** {generated_on}\n"
        "\n"
        "---"
    )
    return cover


REQUIRED_SECTIONS = [
    "## Introduction",
    "## Key Findings",
    "## Challenges",
    "## Future Scope",
    "## Conclusion",
]


def normalize_report_sections(report_body: str) -> str:
    """Ensure each required section header exists in the report body.

    If a section is missing, an empty placeholder is appended so the
    output always has a consistent structure.
    """
    normalized = report_body.strip()

    for section in REQUIRED_SECTIONS:
        # Check case-insensitively
        pattern = re.compile(re.escape(section), re.IGNORECASE)
        if not pattern.search(normalized):
            normalized += f"\n\n{section}\n\n_No data available for this section._"

    return normalized


def assemble_final_report(
    topic: str,
    report_body: str,
    author: str = "Autonomous Research Agent",
) -> str:
    """Combine the cover page with the normalized report body."""
    cover = build_cover_page(topic, author)
    body = normalize_report_sections(report_body)
    return f"{cover}\n\n{body}\n"
