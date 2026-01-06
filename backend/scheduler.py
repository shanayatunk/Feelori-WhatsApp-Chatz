# /backend/scheduler.py

import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config.settings import settings
from app.utils.tasks import update_escalation_analytics, process_abandoned_checkouts, refresh_visual_search_index
from app.jobs.abandoned_cart_nudge_sender import send_abandoned_cart_nudges

# Configure basic logging for this service
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger("SchedulerService")

async def main():
    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")

    # Job 1: Update the escalation dashboard analytics every 5 minutes
    scheduler.add_job(
        update_escalation_analytics,
        'interval',
        minutes=5,
        id="update_escalation_analytics_job"
    )
    logger.info("Scheduled job: update_escalation_analytics (every 5 minutes).")

    # Job 2: Process abandoned checkouts every 15 minutes
    scheduler.add_job(
        process_abandoned_checkouts,
        'interval',
        minutes=15,
        id="abandoned_checkout_processor_job"
    )
    logger.info("Scheduled job: process_abandoned_checkouts (every 15 minutes).")
    
    # Job 3 (Conditional): Refresh the visual search index daily
    if settings.VISUAL_SEARCH_ENABLED:
        scheduler.add_job(
            refresh_visual_search_index, 
            'cron', 
            hour=3, 
            minute=0, 
            id="daily_rebuild_index_job"
        )
        logger.info("Scheduled job: refresh_visual_search_index (daily at 3 AM).")
    
    # --- PHASE 4.5: WhatsApp Abandoned Cart Recovery ---
    async def run_abandoned_cart_job():
        from datetime import datetime
        logger.info("Starting scheduled abandoned cart recovery check...")
        # Pass fresh timestamp for accurate elapsed-time calculation
        await send_abandoned_cart_nudges(datetime.utcnow())

    scheduler.add_job(
        run_abandoned_cart_job,
        'interval',
        minutes=30,
        id="abandoned_cart_recovery_job",
        replace_existing=True
    )

    logger.info("Scheduled job: run_abandoned_cart_job (every 30 minutes).")
    # ---------------------------------------------------
    
    scheduler.start()
    logger.info("Scheduler started successfully. Press Ctrl+C to exit.")

    # This loop keeps the script running forever
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()

if __name__ == "__main__":
    asyncio.run(main())