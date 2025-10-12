# /app/config/settings.py

import sys
import re
import base64  # Import the base64 library
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # MongoDB
    mongo_atlas_uri: str
    max_pool_size: int = 10
    min_pool_size: int = 1
    mongo_ssl: bool = True

    # WhatsApp
    whatsapp_access_token: str
    whatsapp_phone_id: str
    whatsapp_verify_token: str
    whatsapp_app_secret: str
    whatsapp_catalog_id: str | None = None
    whatsapp_business_account_id: str | None = None
    whatsapp_webhook_secret: str | None = None

    # Shopify
    shopify_store_url: str = "feelori.myshopify.com"
    shopify_access_token: str
    shopify_webhook_secret: str | None = None
    shopify_storefront_access_token: str | None = None
    product_search_source: str = "storefront"

    # AI APIs
    gemini_api_key: str | None = None
    openai_api_key: str | None = None

    # Doppler and Webhook Secrets
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
    # Default to production for safety. Must be explicitly set to 'development' to enable debug features.
    environment: str = Field(default="production", env="ENVIRONMENT")

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_ssl: bool = False

    # CORS & Hosts
    cors_allowed_origins: str = Field(default="https://feelori.com,https://admin.feelori.com", env="CORS_ALLOWED_ORIGINS")
    allowed_hosts: str = Field(default="feelori.com,*.feelori.com", env="ALLOWED_HOSTS")

    # Observability & Alerting
    sentry_dsn: str | None = None
    sentry_environment: str = "production"
    alerting_webhook_url: str | None = None
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831

    # App Metadata & Limits
    api_version: str = "v1"
    rate_limit_per_minute: int = 100
    auth_rate_limit_per_minute: int = 5

    @field_validator('jwt_secret_key', 'session_secret_key')
    @classmethod
    def key_length_must_be_sufficient(cls, v):
        if len(v) < 32:
            raise ValueError("JWT/Session secret keys must be at least 32 characters long")
        return v

    # --- THIS VALIDATOR IS REPLACED ---
    @field_validator('admin_password')
    @classmethod
    def decode_password_from_base64_and_validate(cls, v: str) -> str:
        """
        Decodes the admin password from Base64 and validates its length.
        The password in the .env/Doppler MUST be Base64 encoded.
        """
        try:
            # Decode the Base64 string from the environment variable
            decoded_password = base64.b64decode(v).decode('utf-8')
        except Exception:
            raise ValueError("ADMIN_PASSWORD is not valid Base64. Please encode your password hash.")
        
        # Now, validate the length of the DECODED password hash
        if len(decoded_password) < 12:
            raise ValueError("The decoded ADMIN_PASSWORD must be at least 12 characters long.")
        
        # Return the original, decoded hash for the application to use
        return decoded_password
    
    @field_validator('whatsapp_phone_id')
    @classmethod
    def phone_id_must_be_digits(cls, v):
        if not re.match(r'^\d+$', v):
            raise ValueError("WHATSAPP_PHONE_ID must contain only digits")
        return v

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

def validate_environment(settings_obj: Settings):
    try:
        if not settings_obj.whatsapp_verify_token:
            raise ValueError("WHATSAPP_VERIFY_TOKEN is required")
        if not settings_obj.gemini_api_key and not settings_obj.openai_api_key:
            raise ValueError("At least one AI API key (GEMINI_API_KEY or OPENAI_API_KEY) must be provided")
        
        required_vars_prod = ['MONGO_ATLAS_URI', 'WHATSAPP_ACCESS_TOKEN', 'SHOPIFY_ACCESS_TOKEN']
        if settings_obj.environment == 'production':
            for var in required_vars_prod:
                if not getattr(settings_obj, var.lower()):
                    raise ValueError(f"{var} is required for production")
        return settings_obj
    except Exception as e:
        print(f"--- [ERROR] Environment validation failed: {str(e)}")
        sys.exit(1)

settings = Settings()
validate_environment(settings)