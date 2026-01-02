# /app/dependencies/tenant.py

from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config.settings import settings

# Setup HTTPBearer for token extraction
security = HTTPBearer()


def get_tenant_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Extract and validate tenant_id from JWT token.
    
    Args:
        credentials: HTTP Authorization credentials containing the Bearer token
        
    Returns:
        tenant_id string from the token payload
        
    Raises:
        HTTPException 401: If token is invalid or expired
        HTTPException 403: If tenant_id is missing in the token payload
    """
    token = credentials.credentials
    
    try:
        # Decode the JWT token
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=["HS256"]
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token verification failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Extract tenant_id from payload
    tenant_id = payload.get("tenant_id")
    
    # Apply defensive whitespace stripping
    if isinstance(tenant_id, str):
        tenant_id = tenant_id.strip()
    
    # Critical: Raise 403 if tenant_id is missing
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant context missing"
        )
    
    return tenant_id

