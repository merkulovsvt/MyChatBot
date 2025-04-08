import aiohttp
import asyncio
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

base_url = "https://www.litres.ru/genre/russkaya-5297/?art_types=text_book&languages=ru"
semaphore = asyncio.Semaphore(20)


async def fetch_page(session, page):
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
        tasks = [fetch_page(session, page) for page in range(1, 1001)]
        all_results = await tqdm_asyncio.gather(*tasks)
        book_urls = list(set([url for sublist in all_results for url in sublist]))
        pd.DataFrame(book_urls).to_csv('book_urls.csv', header=False, index=False)


asyncio.run(main())
