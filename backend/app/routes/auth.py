# /app/routes/auth.py

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.config.settings import settings
from app.models.api import LoginRequest, TokenResponse, APIResponse
from app.utils.dependencies import verify_jwt_token
from app.utils.request_utils import get_remote_address
from app.services import security_service, jwt_service
from app.services.db_service import db_service
from app.utils.rate_limiter import limiter
# Note: The import for ADMIN_PASSWORD_HASH has been removed.

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

@router.post("/login", response_model=TokenResponse)
@limiter.limit(f"{settings.auth_rate_limit_per_minute}/minute")
async def login(request: Request, login_data: LoginRequest):
    client_ip = get_remote_address(request)
    
    if await security_service.login_tracker.is_locked_out(client_ip):
        security_service.auth_attempts_counter.labels(status="lockout", method="password").inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed attempts. Please try again later."
        )
    
    # --- CORRECTED PASSWORD VALIDATION ---
    # This is now a single, secure check against the hashed password from your .env file.
    password_valid = security_service.EnhancedSecurityService.verify_password(
        login_data.password, settings.admin_password
    )
    
    if not password_valid:
        await security_service.login_tracker.record_attempt(client_ip)
        security_service.auth_attempts_counter.labels(status="failure", method="password").inc()
        await db_service.log_security_event("failed_login", client_ip, {"reason": "invalid_password"})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    # Create access token using the global jwt_service instance
    access_token = jwt_service.jwt_service.create_access_token(
        data={"sub": "admin", "type": "access", "ip": client_ip}
    )
    
    security_service.auth_attempts_counter.labels(status="success", method="password").inc()
    await db_service.log_security_event("successful_login", client_ip, {"method": "jwt"})
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.jwt_access_token_expire_hours * 3600
    )

@router.post("/logout", response_model=APIResponse)
@limiter.limit("10/minute")
async def logout(request: Request, current_user: dict = Depends(verify_jwt_token)):
    client_ip = get_remote_address(request)
    await db_service.log_security_event("logout", client_ip, {"user": current_user.get("sub")})
    return APIResponse(success=True, message="Logged out successfully", version=settings.api_version)

@router.get("/me", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def read_current_user(request: Request, current_user: dict = Depends(verify_jwt_token)):
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    user_data = {"username": current_user.get("sub")}
    return APIResponse(
        success=True,
        message="User authenticated successfully.",
        data={"user": user_data},
        version=settings.api_version
    )