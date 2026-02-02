from typing import List, Dict, Any
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# def _whitespace_tokens(text: str) -> List[str]:
#     """Простая токенизация по пробельным разделителям (быстрая ап prox)."""
#     if not text:
#         return []
#     return text.split()
#
#
# def _tokens_to_text(tokens: List[str]) -> str:
#     return " ".join(tokens)
#
#
# def _split_long_text_by_token_count(text: str, max_tokens: int) -> List[str]:
#     """Разбивает длинный текст (single line) на куски по max_tokens, по словам."""
#     toks = _whitespace_tokens(text)
#     if not toks:
#         return []
#     parts = []
#     for i in range(0, len(toks), max_tokens):
#         parts.append(_tokens_to_text(toks[i : i + max_tokens]))
#     return parts
#
#
# def my_count_tokens(string: str, encoding_name="o200k_base"):
#     encoding = tiktoken.get_encoding(encoding_name)
#
#     tokens = encoding.encode(string)
#     token_count = len(tokens)
#
#     return token_count
#
# def my_split_page(page: Dict[str, Any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
#     """
#     Split page['text'] into chunks preserving lines (чтобы не ломать markdown-таблицы).
#     Возвращает список словарей: {"url": page['url'], "length_tokens": n, "text": chunk_text}.
#     Зависит от метода self.count_tokens(text) для точного подсчёта токенов (например, tiktoken).
#     Для формирования overlap используется простая whitespace-токенизация.
#     """
#     text = page.get("text", "") or ""
#     url = page.get("url", "")
#     if not text:
#         return [{"url": url, "length_tokens": 0, "text": ""}]
#
#     lines = text.splitlines()
#     chunks: List[str] = []
#     current_lines: List[str] = []
#
#     def flush_current():
#         """Сформировать чанк из current_lines и добавить в chunks (если не пусто)."""
#         if not current_lines:
#             return
#         chunk_text = "\n".join(current_lines).strip()
#         if chunk_text:
#             chunks.append(chunk_text)
#
#     for line in lines:
#         # попробуем добавить следующую строку в текущий chunk
#         candidate_lines = current_lines + [line]
#         candidate_text = "\n".join(candidate_lines).strip()
#         if not candidate_text:
#             # пустая строка — добавляем и продолжаем
#             current_lines.append(line)
#             continue
#
#         # если влезает — просто добавляем строку
#         if my_count_tokens(candidate_text) <= chunk_size:
#             current_lines.append(line)
#             continue
#
#         # иначе: текущий chunk закрываем (если есть), а строку обрабатываем отдельно
#         if current_lines:
#             flush_current()
#
#             # создаём overlap — последние chunk_overlap токенов последнего чанка
#             last_chunk = chunks[-1]
#             last_toks = _whitespace_tokens(last_chunk)
#             if chunk_overlap > 0 and last_toks:
#                 overlap_toks = last_toks[-chunk_overlap:]
#                 # начинаем новый current_lines с overlap (как одна "строка")
#                 overlap_text = _tokens_to_text(overlap_toks)
#                 current_lines = [overlap_text]
#             else:
#                 current_lines = []
#
#         # теперь строка может быть всё ещё слишком длинной (одна строка > chunk_size)
#         if my_count_tokens(line) <= chunk_size:
#             current_lines.append(line)
#         else:
#             # разрежем длинную строку по словам на куски по chunk_size
#             parts = _split_long_text_by_token_count(line, chunk_size)
#             for i, p in enumerate(parts):
#                 # если есть предыдущий текущий overlap — объеденим с первой частью, иначе просто добавим
#                 if i == 0 and current_lines:
#                     # попробуем приклеить первую часть к текущему overlap и проверить размер
#                     candidate = "\n".join(current_lines + [p]).strip()
#                     if my_count_tokens(candidate) <= chunk_size:
#                         current_lines = [candidate]
#                         continue
#                     else:
#                         # flush и положим p как отдельный chunk
#                         flush_current()
#                         chunks.append(p)
#                 else:
#                     chunks.append(p)
#             current_lines = []
#
#     # в конце — flush оставшегося
#     if current_lines:
#         flush_current()
#
#     # сформируем финальный список словарей с length_tokens
#     result: List[Dict[str, Any]] = []
#     for c in chunks:
#         # считаем токены с помощью метода self.count_tokens
#         token_count = int(my_count_tokens(c))
#         result.append({"url": url, "length_tokens": token_count, "text": c})
#
#     return result

def count_tokens(string: str, encoding_name="o200k_base"):
    encoding = tiktoken.get_encoding(encoding_name)

    tokens = encoding.encode(string)
    token_count = len(tokens)

    return token_count

def split_page(page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, any]]:
    """Split page text into chunks. The original text includes markdown tables."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(page['text'])
    chunks_with_meta = []
    for chunk in chunks:
        chunks_with_meta.append({
            "url": page['url'],
            "length_tokens": count_tokens(chunk),
            "text": chunk
        })
    return chunks_with_meta
