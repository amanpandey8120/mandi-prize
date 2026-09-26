"""
Supabase service — fetches prediction features for a given
state / district / commodity via the get_prediction_features RPC.
"""

import logging
import os
from typing import Any

from supabase import create_client, Client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build the Supabase client once at import time
# ---------------------------------------------------------------------------
_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_client: Client | None = None

if _SUPABASE_URL and _SUPABASE_KEY:
    try:
        _client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
        logger.info("Supabase client initialised successfully.")
    except Exception as exc:
        logger.error("Failed to initialise Supabase client: %s", exc)
else:
    logger.warning(
        "SUPABASE_URL or SUPABASE_KEY not set — Supabase features disabled."
    )

# Required keys that the RPC must return
_REQUIRED_KEYS = [
    "Min_Price",
    "Max_Price",
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_7",
    "rolling_30",
]


def get_prediction_features(
    state: str,
    district: str,
    commodity: str,
) -> dict[str, float]:
    """
    Call the Supabase RPC `get_prediction_features` and return the
    numeric feature dict needed for model inference.

    Raises
    ------
    RuntimeError
        If the Supabase client is not configured.
    ValueError
        If the RPC returns no data or is missing required features.
    """
    if _client is None:
        raise RuntimeError(
            "Supabase client is not configured. "
            "Set SUPABASE_URL and SUPABASE_KEY environment variables."
        )

    try:
        response = _client.rpc(
            "get_prediction_features",
            {"p_state": state, "p_district": district, "p_commodity": commodity},
        ).execute()
    except Exception as exc:
        raise RuntimeError(
            f"Supabase RPC call failed: {exc}"
        ) from exc

    data: Any = response.data

    # RPC may return a list with one row or a dict directly
    if isinstance(data, list):
        if not data:
            raise ValueError(
                f"No mandi data found in Supabase for "
                f"state='{state}', district='{district}', commodity='{commodity}'. "
                "Ensure this combination exists in the mandi_prices table."
            )
        row = data[0]
    elif isinstance(data, dict):
        row = data
    else:
        raise ValueError(
            f"Unexpected Supabase response format: {type(data)}"
        )

    # Validate all required features are present and numeric
    features: dict[str, float] = {}
    missing = []
    for key in _REQUIRED_KEYS:
        val = row.get(key)
        if val is None:
            missing.append(key)
        else:
            try:
                features[key] = float(val)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Feature '{key}' from Supabase has non-numeric value: {val!r}"
                )

    if missing:
        raise ValueError(
            f"Supabase RPC response is missing required features: {missing}. "
            "Check the get_prediction_features function in your Supabase project."
        )

    return features
