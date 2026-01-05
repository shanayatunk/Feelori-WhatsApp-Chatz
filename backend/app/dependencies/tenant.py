# /app/dependencies/tenant.py

from jose import JWTError, jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config.settings import settings
from app.models.config import BusinessConfig

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


async def get_business_config(
    request: Request,
    business_id: str
) -> BusinessConfig:
    """
    Fetch BusinessConfig from MongoDB and attach to request.state.
    Implements per-request caching (not global, not singleton).
    
    Args:
        request: FastAPI Request object for per-request state
        business_id: Business identifier
        
    Returns:
        BusinessConfig instance
        
    Raises:
        HTTPException 404: If BusinessConfig not found in database
        HTTPException 500: If database query fails
    """
    # Per-request caching: Check if already fetched in this request
    if hasattr(request.state, "business_config"):
        return request.state.business_config
    
    try:
        # Fetch BusinessConfig using Beanie ORM
        business_config = await BusinessConfig.find_one(
            BusinessConfig.business_id == business_id
        )
        
        if not business_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"BusinessConfig not found for business_id: {business_id}"
            )
        
        # Cache in request.state for this request only
        request.state.business_config = business_config
        
        return business_config
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch BusinessConfig: {str(e)}"
        )

