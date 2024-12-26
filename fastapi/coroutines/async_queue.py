import asyncio


async def worker(queue):
    while True:
        task = await queue.get()
        # print(len(queue))
        try:
            print(f'Processing task: {task}')
            await asyncio.sleep(1)  # 模拟任务处理
        finally:
            queue.task_done()

async def main():
    queue = asyncio.Queue()

    # 创建并启动工作协程
    workers = [asyncio.create_task(worker(queue)) for _ in range(3)]

    # 将任务放入队列
    for i in range(10):
        await queue.put(i)

    # 等待所有任务完成
    await queue.join()

    # 取消工作协程
    for w in workers:
        w.cancel()

asyncio.run(main())
