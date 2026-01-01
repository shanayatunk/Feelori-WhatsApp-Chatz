# /app/config/settings.py

import sys
import re
import os
import base64
from typing import Dict, List, Union, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class WhatsAppBusinessConfig(BaseModel):
    """
    Configuration model for a WhatsApp business account.
    Stores metadata and references to environment variables for secure token management.
    """
    business_id: str
    business_name: str
    phone_number_id: str
    token_env_key: str  # Name of the environment variable containing the access token (NOT the token itself)

    class Config:
        frozen = True


class BusinessConfig(BaseModel):
    """
    Configuration model for business metadata (contact info, address, etc.).
    Used for business information display and customer support.
    """
    business_name: str
    support_email: str
    support_phone: str
    website_url: str
    business_address: str


class Settings(BaseSettings):
    # MongoDB
    mongo_atlas_uri: str
    max_pool_size: int = 10
    min_pool_size: int = 1
    mongo_ssl: bool = True

    # --- WhatsApp Credentials (Multi-Tenant) ---
    # Default / Feelori
    whatsapp_phone_id: str = Field(..., env="WHATSAPP_PHONE_ID")
    whatsapp_access_token: str = Field(..., env="WHATSAPP_ACCESS_TOKEN")
    whatsapp_catalog_id: Optional[str] = Field(None, env="WHATSAPP_CATALOG_ID")
    whatsapp_app_secret: str = Field(..., env="WHATSAPP_APP_SECRET")
    whatsapp_business_account_id: Optional[str] = Field(None, env="WHATSAPP_BUSINESS_ACCOUNT_ID")
    whatsapp_verify_token: str
    whatsapp_webhook_secret: str | None = None

    # Golden Collections
    golden_whatsapp_phone_id: Optional[str] = Field(None, env="GOLDEN_WHATSAPP_PHONE_ID")
    golden_whatsapp_access_token: Optional[str] = Field(None, env="GOLDEN_WHATSAPP_ACCESS_TOKEN")
    golden_whatsapp_catalog_id: Optional[str] = Field(None, env="GOLDEN_WHATSAPP_CATALOG_ID")
    golden_whatsapp_app_secret: str = Field("", env="GOLDEN_WHATSAPP_APP_SECRET")
    golden_whatsapp_business_account_id: Optional[str] = Field(None, env="GOLDEN_WHATSAPP_BUSINESS_ACCOUNT_ID")

    # Multi-tenant WhatsApp Business Registry
    WHATSAPP_BUSINESS_REGISTRY: Dict[str, WhatsAppBusinessConfig] = {}

    # Shopify
    shopify_store_url: str = "feelori.myshopify.com"
    shopify_access_token: str
    shopify_webhook_secret: str | None = None
    shopify_storefront_access_token: str | None = None
    product_search_source: str = "storefront"

    # AI APIs
    gemini_api_key: str | None = None
    openai_api_key: str | None = None

    # Doppler / Infra
    doppler_config: str | None = None
    doppler_environment: str | None = None
    doppler_project: str | None = None
    next_public_api_url: str | None = None

    # App Behavior
    packing_dept_whatsapp_number: str | None = None
    packing_executive_names: str = "Swathi,Dharam,Pushpa,Deepika"
    dashboard_url: str = "https://example.com/static/dashboard.html"
    VISUAL_SEARCH_ENABLED: bool = False

    # Security
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_hours: int = 24
    admin_password: str
    session_secret_key: str
    api_key: str | None = None

    # Deployment
    https_only: bool = False
    ssl_cert_path: str | None = None
    ssl_key_path: str | None = None
    workers: int = 4
    environment: str = Field(default="production", env="ENVIRONMENT")

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_ssl: bool = False

    # ✅ CORS Configuration
    cors_allowed_origins: Union[List[str], str] = Field(
        default=[
            "https://feelori.com",
            "https://admin.feelori.com",
            "https://messenger.feelori.com",
        ],
        env="CORS_ALLOWED_ORIGINS",
    )

    allowed_hosts: str = Field(
        default="feelori.com,*.feelori.com",
        env="ALLOWED_HOSTS"
    )

    # Observability
    sentry_dsn: str | None = None
    sentry_environment: str = "production"
    alerting_webhook_url: str | None = None
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831

    # App Metadata & Limits
    api_version: str = "v1"
    rate_limit_per_minute: int = 100
    auth_rate_limit_per_minute: int = 5

    # ---------------- Validators ---------------- #

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("jwt_secret_key", "session_secret_key")
    @classmethod
    def key_length_must_be_sufficient(cls, v):
        if len(v) < 32:
            raise ValueError("JWT/Session secret keys must be at least 32 characters long")
        return v

    @field_validator("admin_password")
    @classmethod
    def decode_password_from_base64_and_validate(cls, v: str) -> str:
        try:
            decoded = base64.b64decode(v).decode("utf-8")
        except Exception:
            raise ValueError("ADMIN_PASSWORD is not valid Base64")

        if len(decoded) < 12:
            raise ValueError("Decoded ADMIN_PASSWORD must be at least 12 characters")

        return decoded

    @field_validator("whatsapp_phone_id")
    @classmethod
    def phone_id_must_be_digits(cls, v):
        if not re.match(r"^\d+$", v):
            raise ValueError("WHATSAPP_PHONE_ID must contain only digits")
        return v

    @model_validator(mode="after")
    def initialize_whatsapp_business_registry(self):
        if not self.WHATSAPP_BUSINESS_REGISTRY and self.whatsapp_phone_id:
            self.WHATSAPP_BUSINESS_REGISTRY[self.whatsapp_phone_id] = WhatsAppBusinessConfig(
                business_id="feelori",
                business_name="Feelori",
                phone_number_id=self.whatsapp_phone_id,
                token_env_key="WHATSAPP_ACCESS_TOKEN"
            )
        return self

    def get_business_token(self, business_config: WhatsAppBusinessConfig) -> str:
        token = os.getenv(business_config.token_env_key)
        if not token:
            raise RuntimeError(
                f"Missing WhatsApp token for business_id={business_config.business_id}"
            )
        return token

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Business Configuration Registry
BUSINESS_REGISTRY: Dict[str, BusinessConfig] = {
    "feelori": BusinessConfig(
        business_name="FeelOri",
        support_email="support@feelori.com",
        support_phone="+91 9967680579",
        website_url="https://feelori.com",
        business_address="Sai Nidhi, Plot 9, Krishnapuri Colony, West Marredpally, Hyderabad, Telangana 500026, India"
    ),
    "goldencollections": BusinessConfig(
        business_name="Golden Collections",
        support_email="support@goldencollections.com",
        support_phone="+91 9967680579",  # TODO: Update with correct Golden Collections phone number
        website_url="https://goldencollections.com",
        business_address="Sai Nidhi, Plot 9, Krishnapuri Colony, West Marredpally, Hyderabad, Telangana 500026, India"  # TODO: Update with Golden Collections specific address if different
    )
}


def get_business_config(business_id: str) -> BusinessConfig:
    """
    Returns the business configuration for the given business_id.
    Defaults to 'feelori' if the business_id is not found in the registry.
    
    Args:
        business_id: The business identifier (e.g., 'feelori', 'goldencollections')
        
    Returns:
        BusinessConfig instance for the requested business
    """
    normalized_id = (business_id or "feelori").strip().lower()
    return BUSINESS_REGISTRY.get(normalized_id, BUSINESS_REGISTRY["feelori"])


def validate_environment(settings_obj: Settings):
    try:
        if not settings_obj.whatsapp_verify_token:
            raise ValueError("WHATSAPP_VERIFY_TOKEN is required")

        if not settings_obj.gemini_api_key and not settings_obj.openai_api_key:
            raise ValueError("At least one AI API key must be provided")

        if settings_obj.environment == "production":
            for var in ["mongo_atlas_uri", "whatsapp_access_token", "shopify_access_token"]:
                if not getattr(settings_obj, var):
                    raise ValueError(f"{var.upper()} is required in production")

        return settings_obj

    except Exception as e:
        print(f"--- [ERROR] Environment validation failed: {e}")
        sys.exit(1)


settings = Settings()
validate_environment(settings)
