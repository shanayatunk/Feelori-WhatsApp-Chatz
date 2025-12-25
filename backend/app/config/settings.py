# /app/config/settings.py

import sys
import re
import os
import base64
from typing import Dict, List
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class BusinessConfig(BaseModel):
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


class Settings(BaseSettings):
    # MongoDB
    mongo_atlas_uri: str
    max_pool_size: int = 10
    min_pool_size: int = 1
    mongo_ssl: bool = True

    # WhatsApp (Legacy – kept for backward compatibility)
    whatsapp_access_token: str  # @deprecated
    whatsapp_phone_id: str      # @deprecated
    whatsapp_verify_token: str
    whatsapp_app_secret: str
    whatsapp_catalog_id: str | None = None
    whatsapp_business_account_id: str | None = None
    whatsapp_webhook_secret: str | None = None

    # Multi-tenant WhatsApp Business Registry
    BUSINESS_REGISTRY: Dict[str, BusinessConfig] = {}

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

    # ✅ CORS — FIXED (messenger.feelori.com added)
    cors_allowed_origins: List[str] = Field(
        default=[
            "https://feelori.com",
            "https://admin.feelori.com",
            "https://message-whisperer-dash.lovable.app",
            "https://messenger.feelori.com"
        ],
        env="CORS_FORCE_DEFAULT"
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
    def parse_cors_allowed_origins(cls, v):
        """
        Handle both string (comma-separated) and list formats for cors_allowed_origins.
        This allows backward compatibility with environment variables that use comma-separated strings.
        """
        if isinstance(v, str):
            # Split by comma and strip whitespace, filter out empty strings
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        elif isinstance(v, list):
            # Already a list, return as is
            return v
        else:
            # For None or other types, return as is (will use default)
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
    def initialize_business_registry(self):
        if not self.BUSINESS_REGISTRY and self.whatsapp_phone_id:
            self.BUSINESS_REGISTRY[self.whatsapp_phone_id] = BusinessConfig(
                business_id="feelori",
                business_name="Feelori",
                phone_number_id=self.whatsapp_phone_id,
                token_env_key="WHATSAPP_ACCESS_TOKEN"
            )
        return self

    def get_business_token(self, business_config: BusinessConfig) -> str:
        token = os.getenv(business_config.token_env_key)
        if not token:
            raise RuntimeError(
                f"Missing WhatsApp token for business_id={business_config.business_id}"
            )
        return token

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


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
