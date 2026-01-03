# backend/fix_db_index.py
import asyncio
from app.services.db_service import db_service

async def fix_index():
    print("🔧 Fixing MongoDB Index for 'wamid'...")
    try:
        # 1. Drop the old strict index
        await db_service.db.message_logs.drop_index("wamid_1")
        print("✅ Dropped old strict index.")
    except Exception as e:
        print(f"⚠️ Could not drop index (might not exist): {e}")

    try:
        # 2. Create the new "Sparse" index (Allows nulls/missing values)
        await db_service.db.message_logs.create_index(
            [("wamid", 1)], 
            unique=True, 
            sparse=True,  # <--- THIS IS THE MAGIC KEY
            background=True
        )
        print("✅ Created new SPARSE index. You can now save messages without IDs!")
    except Exception as e:
        print(f"❌ Failed to create index: {e}")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(fix_index())

