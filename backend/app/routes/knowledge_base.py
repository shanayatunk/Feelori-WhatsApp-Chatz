# /app/routes/knowledge_base.py

import logging
from fastapi import APIRouter, Depends, HTTPException
from app.models.config import KnowledgeBase
from app.utils.dependencies import verify_jwt_token
from app.services.db_service import db_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/knowledge-base",
    tags=["Knowledge Base"],
    dependencies=[Depends(verify_jwt_token)]
)

# Alias for API compatibility
KnowledgeBaseConfig = KnowledgeBase

@router.patch("/{business_id}")
async def update_knowledge_base(business_id: str, payload: KnowledgeBaseConfig, user=Depends(verify_jwt_token)):
    """
    Update the knowledge base for a specific business.
    The payload is the *content* of the KB, so we must wrap it in "knowledge_base" before saving.
    """
    try:
        # 1. Convert Pydantic model to dict
        kb_content = payload.model_dump(mode='json')
        
        # 2. Update DB with correct nesting
        # We use specific fields to avoid overwriting 'persona' or 'rules'
        result = await db_service.db.business_configs.update_one(
            {"business_id": business_id},
            {"$set": {"knowledge_base": kb_content}},  # <--- THE FIX: Nested under 'knowledge_base'
            upsert=True
        )
        
        if result.matched_count == 0 and result.upserted_id:
            logger.info(f"Created new business config for {business_id} with knowledge base")
        else:
            logger.info(f"Updated knowledge base for {business_id}")
        
        return {"status": "success", "message": "Knowledge base updated"}
    except Exception as e:
        logger.error(f"Error updating knowledge base for {business_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update knowledge base: {str(e)}")

@router.get("/{business_id}")
async def get_knowledge_base(business_id: str, user=Depends(verify_jwt_token)):
    """
    Get the knowledge base for a specific business.
    """
    try:
        config = await db_service.db.business_configs.find_one({"business_id": business_id})
        if not config:
            raise HTTPException(status_code=404, detail="Business config not found")
        
        kb = config.get("knowledge_base", {})
        return {"status": "success", "data": kb}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching knowledge base for {business_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch knowledge base: {str(e)}")
