import re
import json
import aiohttp
import asyncio
import pandas as pd
from html import unescape
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

book_urls = pd.read_csv('book_urls.csv', header=None, index_col=False)[0].unique()

title_selector = 'h1[itemprop="name"]'
author_selector = '.Truncate_truncated__jKdVt'
read_more_selector = 'Truncate_readMore__F_GTG'
description_selector = r'\\"html_annotation\\":\\"(.*?)(?<!\\)\\"'

sem = asyncio.Semaphore(20)


async def fetch(session, url):
    async with sem:
        try:
            async with session.get(url) as response:
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')

                title = soup.select_one(title_selector).text
                author = soup.select_one(author_selector).text

                script = soup.find('script', {'id': '__NEXT_DATA__'})
                match = re.search(description_selector, script.text)
                escaped_json_str = '{"temp": "' + match.group(1) + '"}'
                decoded_data = json.loads(escaped_json_str)
                description = BeautifulSoup(unescape(decoded_data['temp']), 'html.parser').get_text()

                return [author, title, str(description)]
        except Exception as e:
            return None


async def main():
    data = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in book_urls]
        for result in tqdm_asyncio.as_completed(tasks):
            item = await result
            if item:
                data.append(item)
    return data


data = asyncio.run(main())


def clean_text(text):
    text = text.replace('\\n', ' ')
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


parsed_data = pd.DataFrame(data, columns=['author', 'title', 'description'])
parsed_data['description'] = parsed_data['description'].apply(clean_text)

final_data = parsed_data.apply(
    lambda
        row: f"Автор - {row['author']}.\n Название книги - {row['title']}.\n Описание книги - {row['description']}",
    axis=1)
final_data.to_csv('parsed_book_data.txt', header=None, index=None, sep=' ', mode='a')
