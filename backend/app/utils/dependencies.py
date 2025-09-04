# /app/utils/dependencies.py

import base64
import hmac
import hashlib
import secrets
from fastapi import Request, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.config.settings import settings
from app.services import jwt_service, security_service
from app.services.db_service import db_service
from app.utils.metrics import webhook_signature_counter
from app.utils.request_utils import get_remote_address

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"/api/{settings.api_version}/auth/login")

async def verify_jwt_token(token: str = Depends(oauth2_scheme)) -> dict:
    # FIX: Use the jwt_service instance, not the module
    payload = jwt_service.jwt_service.verify_token(token)
    if payload.get("sub") != "admin" or payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return payload

async def verify_webhook_signature(request: Request):
    body = await request.body()
    signature = request.headers.get("x-hub-signature-256", "")
    if not security_service.SecurityService.verify_webhook_signature(body, signature, settings.whatsapp_app_secret):
        webhook_signature_counter.labels(status="invalid").inc()
        await db_service.log_security_event("invalid_webhook_signature", get_remote_address(request), {"signature": signature[:50]})
        raise HTTPException(status_code=403, detail="Invalid signature")
    webhook_signature_counter.labels(status="valid").inc()
    return body

async def verify_shopify_signature(request: Request) -> bytes:
    if not settings.shopify_webhook_secret:
        raise HTTPException(status_code=501, detail="Shopify webhook processing is not configured.")
    
    body = await request.body()
    header_val = request.headers.get("X-Shopify-Hmac-Sha256", "")
    if not header_val:
        raise HTTPException(status_code=403, detail="Missing HMAC header")
        
    expected = base64.b64encode(hmac.new(settings.shopify_webhook_secret.encode(), body, hashlib.sha256).digest()).decode()
    if not hmac.compare_digest(expected, header_val):
        await db_service.log_security_event(
            "invalid_shopify_signature",
            get_remote_address(request),
            {"header_prefix": header_val[:20]}
        )
        raise HTTPException(status_code=403, detail="Invalid Shopify signature")
    return body

async def verify_metrics_access(request: Request):
    if settings.api_key:
        provided_key = request.headers.get("x-api-key") or ""
        if not secrets.compare_digest(provided_key, settings.api_key):
            raise HTTPException(status_code=403, detail="Invalid API key")
    return True