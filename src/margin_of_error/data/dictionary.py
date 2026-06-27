"""Parser for the Ames data description file.

Parses data_description.txt (from the Kaggle competition or De Cock's original)
into a structured reference that can be used for:
  - Validating allowed categorical values in pandera schemas
  - Generating human-readable feature documentation
  - Mapping ordinal quality codes (Ex/Gd/TA/Fa/Po) to integers

The description file format is:
    ColumnName: Brief description of the feature.

           CODE    Description of code
           CODE    Description of code
           ...

Empty lines separate features. Indented lines are code definitions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeatureDefinition:
    """Structured definition for a single feature from the data dictionary.

    Attributes:
        name: Column name as it appears in the dataset.
        description: Human-readable feature description.
        allowed_values: Mapping of code → description for categorical features.
            Empty for numeric features.
        is_categorical: True if the feature has an enumerated set of codes.
    """

    name: str
    description: str
    allowed_values: dict[str, str] = field(default_factory=dict)

    @property
    def is_categorical(self) -> bool:
        return bool(self.allowed_values)

    @property
    def allowed_codes(self) -> list[str]:
        """Return the list of valid codes as strings."""
        return list(self.allowed_values.keys())


DataDictionary = dict[str, FeatureDefinition]

# Pattern for a code line: 2+ leading whitespace, a code token, then a separator
# (tab(s) or 2+ spaces — the real description file uses spaces; some conversions use tabs),
# then the description text.
_CODE_PATTERN = re.compile(r"^\s{2,}(\S.*?)(?:\t+|\s{2,})(.+)$")
# Pattern for a feature header: "ColumnName: Description" (no leading whitespace)
_HEADER_PATTERN = re.compile(r"^(\w+)\s*:\s*(.*)$")


def parse_description_file(path: Path | str) -> DataDictionary:
    """Parse data_description.txt into a structured DataDictionary.

    Args:
        path: Path to the description file.

    Returns:
        Dict mapping column name → FeatureDefinition.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or no features could be parsed.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    dictionary: DataDictionary = {}
    current_feature: FeatureDefinition | None = None

    for line in lines:
        # Skip blank lines
        if not line.strip():
            continue

        # Try to match a feature header (non-indented "ColumnName: ...")
        header_match = _HEADER_PATTERN.match(line)
        if header_match and not line.startswith(" ") and not line.startswith("\t"):
            name, description = header_match.group(1).strip(), header_match.group(2).strip()
            current_feature = FeatureDefinition(name=name, description=description)
            dictionary[name] = current_feature
            continue

        # Try to match a code definition (indented "   CODE    Description")
        code_match = _CODE_PATTERN.match(line)
        if code_match and current_feature is not None:
            code = code_match.group(1).strip()
            desc = code_match.group(2).strip()
            current_feature.allowed_values[code] = desc
            continue

    if not dictionary:
        raise ValueError(f"No features parsed from {path}. Check file format.")

    return dictionary


def get_quality_ordinal_map() -> dict[str, int]:
    """Return the standard ordinal mapping for quality/condition codes.

    Used to convert string quality codes to integers for ordinal features
    (ExterQual, KitchenQual, BsmtQual, etc.).

    Returns:
        Dict mapping code → integer (higher = better).
    """
    return {
        "Ex": 5,
        "Gd": 4,
        "TA": 3,
        "Fa": 2,
        "Po": 1,
        "NA": 0,  # "Not Applicable" (e.g., no basement → BsmtQual = NA)
    }


def summarize_dictionary(dictionary: DataDictionary) -> str:
    """Return a human-readable summary of the data dictionary.

    Args:
        dictionary: Parsed DataDictionary.

    Returns:
        Multi-line string with feature counts and examples.
    """
    categorical = [f for f in dictionary.values() if f.is_categorical]
    numeric = [f for f in dictionary.values() if not f.is_categorical]
    lines = [
        f"Data dictionary: {len(dictionary)} features",
        f"  Categorical: {len(categorical)}",
        f"  Numeric/other: {len(numeric)}",
    ]
    return "\n".join(lines)
