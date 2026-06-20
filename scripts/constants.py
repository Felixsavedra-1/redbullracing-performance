import datetime

DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = datetime.date.today().year
DNF_POSITION_ORDER = 999

try:
    from config import TEAM_CONFIG  # type: ignore[import]
    TEAM_REFS: list[str] = TEAM_CONFIG["family_refs"]
    TEAM_NAME: str = TEAM_CONFIG["name"]
    TEAM_COLORS: dict = TEAM_CONFIG.get("colors", {})
except ImportError:
    TEAM_REFS = ["red_bull"]
    TEAM_NAME = "Oracle Red Bull Racing"
    TEAM_COLORS = {}
except KeyError as e:
    raise KeyError(f"config.py is present but missing required key: {e}") from e

CONSTRUCTOR_ID = 9

TEAM_COLORS.setdefault("primary", "#1E41FF")
TEAM_COLORS.setdefault("accent",  "#FF1800")
TEAM_COLORS.setdefault("neutral", "#AAAAAA")
