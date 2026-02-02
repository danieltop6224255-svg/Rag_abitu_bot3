import trafilatura
from typing import List, Dict, Optional
import time
import logging

import trafilatura


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Parser:
    """
    Parser — парсит список URL и возвращает список словарей: {"url": ..., "text": ...}
    Параметры:
        user_agent: строка User-Agent (по умолчанию простая кастомная строка)
        crawl_delay: задержка (сек) между запросами (политный режим)
        include_tables: передаётся в trafilatura.extract (если нужно сохранять таблицы)
        output_format: 'plain' или 'markdown' — прокидывается в trafilatura.extract
    """
    def __init__(
        self,
        user_agent: Optional[str] = None,
        crawl_delay: float = 0.5,
        include_tables: bool = True,
        output_format: str = "plain",
    ):
        self.user_agent = user_agent or "my-scraper-bot/1.0 (+https://example.com)"
        self.crawl_delay = crawl_delay
        self.include_tables = include_tables
        self.output_format = output_format if output_format in ("plain", "markdown") else "plain"

    def _extract_with_trafilatura(self, html: str) -> Optional[str]:
        if not html:
            return None
        return trafilatura.extract(
            html,
            include_comments=False,
            include_tables=self.include_tables,
            output_format=self.output_format
        )

    def _fetch_with_trafilatura(self, url: str) -> Optional[str]:
        # trafilatura.fetch_url не принимает timeout в некоторых версиях
        try:
            downloaded = trafilatura.fetch_url(url)
            return downloaded
        except Exception as e:
            logger.debug("trafilatura.fetch_url error for %s: %s", url, e)
            return None

    def parse_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Принимает список URL, возвращает список словарей {"url": url, "text": extracted_text}.
        Если не удалось извлечь текст — возвращается пустая строка.
        """
        results: List[Dict[str, str]] = []

        for i, url in enumerate(urls):
            logger.info("Parsing %d/%d: %s", i + 1, len(urls), url)
            time.sleep(self.crawl_delay)  # polite delay

            text = ""

            # 1) Попытка: trafilatura.fetch_url()
            downloaded = self._fetch_with_trafilatura(url)
            if downloaded:
                logger.debug("trafilatura.fetch_url вернул HTML для %s", url)
                extracted = self._extract_with_trafilatura(downloaded)
                if extracted:
                    text = extracted

            if not text:
                logger.warning("Не удалось извлечь текст с %s (оставляю пустую строку)", url)

            results.append({"url": url, "text": text or ""})

        return results

