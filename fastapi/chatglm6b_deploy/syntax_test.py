import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def simple_lifespan():
    print("Resource setup")
    yield
    print("Resource cleanup")


async def main():
    async with simple_lifespan():
        print("Inside the context")
        await asyncio.sleep(2)


# 运行示例
asyncio.run(main())
