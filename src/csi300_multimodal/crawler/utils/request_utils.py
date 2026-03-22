from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger(__name__)


def _looks_like_remote(url: str) -> bool:
    return urlparse(str(url)).scheme in {"http", "https"}


def _resolve_local_path(resource: str, base_dir: Path | None = None) -> Path:
    parsed = urlparse(str(resource))
    if parsed.scheme == "file":
        return Path(parsed.path).expanduser().resolve()
    path = Path(resource).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    return path


def resolve_resource(resource: str, base_dir: Path | None = None) -> str:
    if _looks_like_remote(resource) or urlparse(str(resource)).scheme == "file":
        return str(resource)
    return str(_resolve_local_path(resource, base_dir=base_dir))


def fetch_text(
    resource: str,
    *,
    timeout_seconds: int = 15,
    retry_attempts: int = 3,
    base_dir: Path | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    def _read() -> str:
        if _looks_like_remote(resource):
            import requests

            response = requests.get(resource, timeout=timeout_seconds, headers=headers)
            response.raise_for_status()
            response.encoding = response.encoding or response.apparent_encoding or "utf-8"
            return response.text

        path = _resolve_local_path(resource, base_dir=base_dir)
        LOGGER.debug("Reading local crawler fixture: %s", path)
        return path.read_text(encoding="utf-8")

    retryer = Retrying(
        stop=stop_after_attempt(max(1, int(retry_attempts))),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    for attempt in retryer:
        with attempt:
            return _read()
    raise RuntimeError("unreachable")
