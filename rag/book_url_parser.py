import aiohttp
import asyncio
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

urls = [
    "https://www.litres.ru/genre/russkaya-5297/?art_types=text_book&languages=ru",  # Русская классика
    "https://www.litres.ru/popular/?art_types=text_book&languages=ru",  # Популярное
    "https://www.litres.ru/search/?q=Виктор+Пелевин&art_types=text_book&languages=ru"  # Виктор Пелевин
]
semaphore = asyncio.Semaphore(30)


async def fetch_page(session, page, base_url):
    url = base_url if page == 1 else f"{base_url}&page={page}"
    try:
        async with semaphore:
            async with session.get(url, timeout=10) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                links = [
                    f"https://www.litres.ru{a.get('href')}"
                    for a in soup.find_all('a', class_='Art_content__link___s713')
                    if a.get('href')
                ]
                return links

    except Exception as e:
        return []


async def main():
    async with aiohttp.ClientSession() as session:
        print('Классическая Русская литература')
        tasks = [fetch_page(session, page, urls[0]) for page in range(1, 1001)]
        classic_results = await tqdm_asyncio.gather(*tasks)

        print('Популярное')
        tasks = [fetch_page(session, page, urls[1]) for page in range(1, 1001)]
        popular_results = await tqdm_asyncio.gather(*tasks)

        print('Виктор Пелевин')
        tasks = [fetch_page(session, page, urls[2]) for page in range(1, 6)]
        vp_results = await tqdm_asyncio.gather(*tasks)

        all_results = classic_results + popular_results + vp_results

        book_urls = list(set([url for sublist in all_results for url in sublist]))
        pd.DataFrame(book_urls).to_csv('book_urls.csv', header=False, index=False)


asyncio.run(main())
