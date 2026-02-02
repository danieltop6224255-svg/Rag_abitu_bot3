import os
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
from typing import List
import textwrap
import time

load_dotenv()

proxy_pass = os.getenv("PROXY_PASSWORD")
if not proxy_pass:
    raise RuntimeError("PROXY_PASSWORD missing")

os.environ['HTTP_PROXY']  = f'http://proxyuser:{proxy_pass}@72.56.68.7:3128'
os.environ['HTTPS_PROXY'] = f'http://proxyuser:{proxy_pass}@72.56.68.7:3128'



api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY missing")

client = OpenAI(api_key=api_key)

def get_model_answer(question: str, info: List[str], model: str = "gpt-4o") -> str:
    """
    Формирует prompt из question + info и запрашивает ответ у модели OpenAI.
    Возвращает строковый ответ.
    Важные свойства:
      - модель обязана опираться ТОЛЬКО на переданные info (chunks).
      - если info недостаточно для точного ответа — модель должна прямо сказать об этом.
      - модель должна указать, какие чанки (по индексам) были использованы.
    """
    MAX_CONTEXT_CHARS = 12000  # максимум символов суммарно из info (примерная граница)
    selected = []
    total_chars = 0
    for idx, chunk in list(enumerate(info, start=1)):
        chunk_repr = f"[{idx}] {chunk}"
        if total_chars + len(chunk_repr) > MAX_CONTEXT_CHARS:
            break
        selected.insert(0, (idx, chunk))  # вставляем в начало, чтобы сохранить порядок
        total_chars += len(chunk_repr)

    # Если ничего не попало (слишком большие чанки), возьмём последний один короткий кусок (если есть)
    if not selected and info:
        selected = [(1, info[0])]

    # Формируем блок контекста для передачи в модель
    context_lines = []
    for idx, chunk in selected:
        # обрезаем очень длинные чанки (без сильного повреждения смысла)
        snippet = chunk if len(chunk) <= 4000 else chunk[:4000] + " ...[truncated]..."
        context_lines.append(f"[{idx}] {snippet}")

    context_text = "\n\n".join(context_lines) if context_lines else "(no context provided)"

    # Система (роль) — строгие инструкции о поведении
    system_prompt = textwrap.dedent("""\
        Ты — точный и осторожный помощник . Твоя задача — ответить на вопрос пользователя,
        опираясь исключительно на предоставленный блок(и) контента (context).
        - Используй ТОЛЬКО информацию, явно содержащуюся в блоках контекста. 
        - Не придумывай факты (не галлюцинируй).
        - Ответь кратко, чётко и по существу. Примеры формата ответа ниже.
    """)

    # Форматируем сообщение для пользователя (включаем контекст и сам вопрос)
    user_prompt = textwrap.dedent(f"""\
        Контекст (ненумерованные блоки ниже соответствуют номерам в квадратных скобках):
        {context_text}

        ---
        Вопрос: {question}

        Пожалуйста, ответь инженером-стилем:
        Дай прямой ответ (в 1-3 предложениях), используя только контекст.
    """)

    # Сделаем сам вызов — с небольшим количеством токенов для ответа
    try:
        # В некоторых версиях SDK метод chat.completions.create существует; в других — responses API.
        # Здесь используем chat.completions.create (универсальный и знакомый интерфейс).
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=800,   # подстрой под ожидаемую длину ответа
        )

        answer = completion.choices[0].message.content
    except Exception as e:
        # простая обработка ошибок: можно улучшить retry/backoff здесь
        return f"Ошибка при запросе к модели: {e}"

    return answer.strip()


def get_embeddings(texts: List[str], model="text-embedding-3-large", dim=3072):
    resp = client.embeddings.create(
        model=model,
        input=texts,
        dimensions=dim,
    )
    embeddings = [item.embedding for item in resp.data]
    return embeddings
