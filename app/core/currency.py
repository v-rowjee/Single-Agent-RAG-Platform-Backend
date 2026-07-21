"""Currency detection and display helpers for dataset-grounded analysis."""

from __future__ import annotations

from collections.abc import Iterable


_CURRENCY_MARKERS = (
    ("GBP", ("gbp", "pound", "sterling", "£")),
    ("EUR", ("eur", "euro", "€")),
    ("USD", ("usd", "dollar", "$")),
    ("CAD", ("cad", "canadian dollar")),
    ("AUD", ("aud", "australian dollar")),
    ("INR", ("inr", "rupee", "₹")),
    ("JPY", ("jpy", "yen", "¥")),
)

_CURRENCY_SYMBOLS = {
    "GBP": "£",
    "EUR": "€",
    "USD": "$",
    "CAD": "CA$",
    "AUD": "A$",
    "INR": "₹",
    "JPY": "¥",
}


def detect_currency(values: Iterable[object]) -> str | None:
    """Return the first recognised ISO code from column names or metadata."""
    text = " ".join(str(value).casefold() for value in values if value is not None)
    for code, markers in _CURRENCY_MARKERS:
        if any(marker in text for marker in markers):
            return code
    return None


def format_currency(value: float, currency: str | None) -> str:
    """Format a dataset value without inventing a currency when it is unknown."""
    symbol = _CURRENCY_SYMBOLS.get((currency or "").upper(), "")
    return f"{symbol}{value:,.2f}"
