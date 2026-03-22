from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup


def parse_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def select_text(soup: BeautifulSoup, selector: str, separator: str = " ") -> str:
    nodes = soup.select(selector)
    if not nodes:
        return ""
    return separator.join(node.get_text(" ", strip=True) for node in nodes).strip()


def select_attr(soup: BeautifulSoup, selector: str, attr: str) -> str:
    node = soup.select_one(selector)
    if node is None:
        return ""
    return str(node.get(attr, "")).strip()


def extract_links(soup: BeautifulSoup, selector: str, base_url: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for node in soup.select(selector):
        href = str(node.get("href", "")).strip()
        if not href:
            continue
        resolved = urljoin(base_url, href)
        if resolved not in seen:
            seen.add(resolved)
            links.append(resolved)
    return links
