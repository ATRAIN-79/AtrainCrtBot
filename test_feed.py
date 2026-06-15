import asyncio, os
from dotenv import load_dotenv
from po_data_feed import PocketOptionFeed

load_dotenv()

async def test():
    feed = PocketOptionFeed(ssid=os.getenv("PO_SSID"), demo=True)
    candles = await feed.fetch_candles("EUR/USD-OTC", "1min", 5)
    print(f"Got {len(candles)} candles")
    print(candles)

asyncio.run(test())