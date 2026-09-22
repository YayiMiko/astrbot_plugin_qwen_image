"""Per-session reference image slots for multi-image editing.

Slots stage up to 3 workspace images (usually marked from older chat
messages) so a later `/qwen edit` can reference them as 图一/图二/图三.
Storage is one small JSON file per session under the plugin data dir.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

_CN_NUM = {"一": 1, "二": 2, "三": 3}
_NUM_CN = {1: "一", 2: "二", 3: "三"}
_MENTION_PATTERNS = (
    re.compile(r"图\s*([一二三123])"),
    re.compile(r"第\s*([一二三123])\s*张"),
    re.compile(r"image\s*([123])", re.IGNORECASE),
)
_TARGET_PATTERNS = (
    re.compile(r"为\s*图\s*([一二三123])"),
    re.compile(
        r"把\s*图\s*([一二三123]).{0,15}(换成|变成|改为|改成|穿上|换上|改掉)"
    ),
)


def find_image_mentions(text: str) -> list[int]:
    """Extract referenced image numbers (图一/图二/图三 ...) from text.

    Args:
        text: User requirement text.

    Returns:
        Sorted unique 1-based image numbers found in the text.
    """
    found: set[int] = set()
    for pattern in _MENTION_PATTERNS:
        for token in pattern.findall(str(text or "")):
            if token.isdigit():
                found.add(int(token))
            elif token in _CN_NUM:
                found.add(_CN_NUM[token])
    return sorted(found)


def detect_target_mention(text: str) -> int | None:
    """Detect an explicitly named edit target (为图N / 把图N...换成...).

    Args:
        text: User requirement text.

    Returns:
        1-based target image number, or None when the request does not
        name an explicit target.
    """
    for pattern in _TARGET_PATTERNS:
        match = pattern.search(str(text or ""))
        if not match:
            continue
        token = match.group(1)
        if token.isdigit():
            return int(token)
        if token in _CN_NUM:
            return _CN_NUM[token]
    return None


def _remap_token(match: re.Match, mapping: dict[int, int], form: str) -> str:
    token = match.group(1)
    old = int(token) if token.isdigit() else _CN_NUM.get(token, 0)
    new = mapping.get(old, old)
    if form == "tu":
        numeral = _NUM_CN.get(new, str(new)) if not token.isdigit() else str(new)
        return f"图{numeral}"
    if form == "di":
        numeral = _NUM_CN.get(new, str(new)) if not token.isdigit() else str(new)
        return f"第{numeral}张"
    return f"image {new}"


def remap_mentions(text: str, mapping: dict[int, int]) -> str:
    """Renumber image mentions after a target-first reorder.

    Args:
        text: User requirement text with old numbering.
        mapping: Old 1-based number to new 1-based number.

    Returns:
        Text with mentions renumbered (mention forms preserved).
    """
    if not mapping:
        return str(text or "")
    result = re.sub(
        r"图\s*([一二三123])",
        lambda match: _remap_token(match, mapping, "tu"),
        str(text or ""),
    )
    result = re.sub(
        r"第\s*([一二三123])\s*张",
        lambda match: _remap_token(match, mapping, "di"),
        result,
    )
    result = re.sub(
        r"image\s*([123])",
        lambda match: _remap_token(match, mapping, "image"),
        result,
        flags=re.IGNORECASE,
    )
    return result


def reorder_target_first(
    paths: list[str], numbers: list[int], prompt: str
) -> tuple[list[str], list[int], str]:
    """Move an explicitly named edit target to the first supply position.

    The Qwen edit graph always treats the first supplied image as the edit
    target, so when the user names another image as the target (e.g.
    为图2角色穿上图1的衣服) the supply order is permuted and the 图N tokens
    in the prompt are renumbered to match the new order.

    Args:
        paths: Resolved image paths in current supply order.
        numbers: User-visible 1-based numbers aligned with `paths`.
        prompt: User requirement text.

    Returns:
        Tuple of reordered paths, reordered numbers, and renumbered prompt.
    """
    count = len(paths)
    if count < 2:
        return paths, numbers, prompt
    target = detect_target_mention(prompt)
    if target is None or target == numbers[0]:
        return paths, numbers, prompt
    try:
        target_index = numbers.index(target)
    except ValueError:
        return paths, numbers, prompt
    order = [target_index] + [
        index for index in range(count) if index != target_index
    ]
    new_paths = [paths[index] for index in order]
    new_numbers = [numbers[index] for index in order]
    # Old user-visible number -> new 1-based supply position.
    mapping = {
        old: new for new, old in enumerate(new_numbers, start=1)
    }
    return new_paths, new_numbers, remap_mentions(prompt, mapping)


def sanitize_session_key(session_id: str) -> str:
    """Make a session id safe for use as a file name.

    Args:
        session_id: Raw AstrBot session id.

    Returns:
        Filesystem-safe key, truncated to 64 characters.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(session_id or "default"))[:64]


def slot_key(session_id: str, sender_id: str) -> str:
    """Build the per-user slot key for a session and sender.

    Slots are scoped to session + sender so each user keeps an independent
    set even in shared group chats.

    Args:
        session_id: AstrBot session id (group id in groups).
        sender_id: AstrBot sender (user) id.

    Returns:
        Slot key string.
    """
    session = re.sub(r"[^A-Za-z0-9_-]", "_", str(session_id or "default"))[:48]
    sender = re.sub(r"[^A-Za-z0-9_-]", "_", str(sender_id or "unknown"))[:48]
    return f"{session}__{sender}"


class ImageSlotStore:
    """Persist staged reference images per chat session."""

    def __init__(
        self,
        slots_dir: Path,
        logger: Any,
        ttl_minutes: int = 30,
        max_slots: int = 3,
    ):
        """Create a slot store.

        Args:
            slots_dir: Directory holding one JSON file per session.
            logger: Logger compatible with AstrBot logger methods.
            ttl_minutes: Slot expiry in minutes. Values <= 0 disable expiry.
            max_slots: Maximum slots per session.
        """
        self.slots_dir = Path(slots_dir)
        self._logger = logger
        self.ttl_minutes = max(0, int(ttl_minutes or 0))
        self.max_slots = max(1, int(max_slots or 1))

    def _slot_path(self, session_id: str) -> Path:
        return self.slots_dir / f"{sanitize_session_key(session_id)}.json"

    def _load_raw(self, session_id: str) -> list[dict[str, Any]]:
        path = self._slot_path(session_id)
        if not path.is_file():
            return []
        try:
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._logger.warning("[qwen] failed to read slot file %s: %s", path, exc)
            return []
        return [item for item in data if isinstance(item, dict)]

    def _write_raw(self, session_id: str, slots: list[dict[str, Any]]) -> None:
        import json

        self.slots_dir.mkdir(parents=True, exist_ok=True)
        self._slot_path(session_id).write_text(
            json.dumps(slots, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _fresh(self, slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = time.time()
        alive: list[dict[str, Any]] = []
        for slot in slots:
            try:
                age_ok = self.ttl_minutes <= 0 or (
                    now - float(slot.get("saved_at", 0))
                    <= self.ttl_minutes * 60
                )
            except (TypeError, ValueError):
                age_ok = False
            if not age_ok:
                continue
            if not Path(str(slot.get("path") or "")).is_file():
                continue
            alive.append(slot)
        return alive

    def read(self, session_id: str) -> list[dict[str, Any]]:
        """Read live slots for a session, dropping expired entries.

        Args:
            session_id: AstrBot session id.

        Returns:
            Live slots in supply order (each with `path` and `saved_at`).
        """
        alive = self._fresh(self._load_raw(session_id))
        return alive

    def save(self, session_id: str, paths: list[str]) -> dict[str, Any]:
        """Append images to a session's slots with FIFO eviction.

        Args:
            session_id: AstrBot session id.
            paths: Saved local image paths to stage.

        Returns:
            Dict with `slots` (live slots after save) and `evicted` count.
        """
        alive = self._fresh(self._load_raw(session_id))
        now = time.time()
        for path in paths or []:
            if not path:
                continue
            if any(slot.get("path") == path for slot in alive):
                continue
            alive.append({"path": str(path), "saved_at": now})
        evicted = max(0, len(alive) - self.max_slots)
        alive = alive[-self.max_slots :]
        self._write_raw(session_id, alive)
        return {"slots": alive, "evicted": evicted}

    def clear(self, session_id: str) -> int:
        """Clear a session's slots.

        Args:
            session_id: AstrBot session id.

        Returns:
            Number of live slots removed.
        """
        alive = self._fresh(self._load_raw(session_id))
        path = self._slot_path(session_id)
        try:
            if path.is_file():
                path.unlink()
        except OSError as exc:
            self._logger.warning("[qwen] failed to clear slot file %s: %s", path, exc)
        return len(alive)
