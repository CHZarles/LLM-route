
from __future__ import annotations

import asyncio
import html.parser
import pathlib
import time
import urllib.parse
from typing import Callable, Iterable

import httpx  # https://github.com/encode/httpx


# URL过滤器类
class UrlFilterer:
    def __init__(
            self,
            allowed_domains: set[str] | None = None,
            allowed_schemes: set[str] | None = None,
            allowed_filetypes: set[str] | None = None,
    ):
        self.allowed_domains = allowed_domains  # 允许的域名
        self.allowed_schemes = allowed_schemes  # 允许的协议
        self.allowed_filetypes = allowed_filetypes  # 允许的文件类型

    # 过滤URL的方法
    def filter_url(self, base: str, url: str) -> str | None:
        url = urllib.parse.urljoin(base, url)  # 解析相对URL
        url, _frag = urllib.parse.urldefrag(url)  # 去除URL中的片段
        parsed = urllib.parse.urlparse(url)  # 解析URL
        if (self.allowed_schemes is not None
                and parsed.scheme not in self.allowed_schemes):
            return None
        if (self.allowed_domains is not None
                and parsed.netloc not in self.allowed_domains):
            return None
        ext = pathlib.Path(parsed.path).suffix  # 获取文件扩展名
        if (self.allowed_filetypes is not None
                and ext not in self.allowed_filetypes):
            return None
        return url

# URL解析器类
class UrlParser(html.parser.HTMLParser):
    def __init__(
            self,
            base: str,
            filter_url: Callable[[str, str], str | None]
    ):
        super().__init__()
        self.base = base  # 基础URL
        self.filter_url = filter_url  # 过滤URL的函数
        self.found_links = set()  # 找到的链接集合

    # 处理开始标签的方法
    def handle_starttag(self, tag: str, attrs):
        # 查找<a href="...">标签
        if tag != "a":
            return

        for attr, url in attrs:
            if attr != "href":
                continue

            if (url := self.filter_url(self.base, url)) is not None:
                self.found_links.add(url)  # 添加找到的链接

# 爬虫类
class Crawler:
    def __init__(
            self,
            client: httpx.AsyncClient,
            urls: Iterable[str],
            filter_url: Callable[[str, str], str | None],
            workers: int = 10,
            limit: int = 25,
    ):
        self.client = client  # HTTP客户端

        self.start_urls = set(urls)  # 起始URL集合
        self.todo = asyncio.Queue()  # 待处理URL队列
        self.seen = set()  # 已处理URL集合
        self.done = set()  # 完成处理URL集合

        self.filter_url = filter_url  # 过滤URL的函数
        self.num_workers = workers  # 工作协程数量
        self.limit = limit  # 爬取URL的限制数量
        self.total = 0  # 已处理URL计数

    # 运行爬虫的方法
    async def run(self):
        await self.on_found_links(self.start_urls)  # 初始化队列
        workers = [
            asyncio.create_task(self.worker())
            for _ in range(self.num_workers)
        ]
        await self.todo.join()

        for worker in workers:
            worker.cancel()

    # 工作协程的方法
    async def worker(self):
        while True:
            try:
                await self.process_one()
            except asyncio.CancelledError:
                return

    # 处理单个URL的方法
    async def process_one(self):
        url = await self.todo.get()
        try:
            await self.crawl(url)
        except Exception as exc:
            # 错误处理
            pass
        finally:
            self.todo.task_done()

    # 爬取单个URL的方法
    async def crawl(self, url: str):

        # 速率限制
        await asyncio.sleep(.1)

        response = await self.client.get(url, follow_redirects=True)

        found_links = await self.parse_links(
            base=str(response.url),
            text=response.text,
        )

        await self.on_found_links(found_links)

        self.done.add(url)

    # 解析链接的方法
    async def parse_links(self, base: str, text: str) -> set[str]:
        parser = UrlParser(base, self.filter_url)
        parser.feed(text)
        return parser.found_links

    # 处理找到的链接的方法
    async def on_found_links(self, urls: set[str]):
        new = urls - self.seen
        self.seen.update(new)

        for url in new:
            await self.put_todo(url)

    # 将URL放入待处理队列的方法
    async def put_todo(self, url: str):
        if self.total >= self.limit:
            return
        self.total += 1
        await self.todo.put(url)

# 主函数
async def main():
    filterer = UrlFilterer(
        allowed_domains={"mcoding.io"},
        allowed_schemes={"http", "https"},
        allowed_filetypes={".html", ".php", ""},
    )

    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        crawler = Crawler(
            client=client,
            urls=["https://mcoding.io/"],
            filter_url=filterer.filter_url,
            workers=5,
            limit=25,
        )
        await crawler.run()
    end = time.perf_counter()

    seen = sorted(crawler.seen)
    print("Results:")
    for url in seen:
        print(url)
    print(f"Crawled: {len(crawler.done)} URLs")
    print(f"Found: {len(seen)} URLs")
    print(f"Done in {end - start:.2f}s")

# 运行主函数
if __name__ == '__main__':
    asyncio.run(main(), debug=True)
