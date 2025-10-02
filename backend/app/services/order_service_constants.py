# /app/services/order_service_constants.py

class CacheKeys:
    """A single source of truth for all cache keys."""
    # --- Customer & State Keys ---
    CUSTOMER_DATA_V2 = "customer:v2:{phone}"
    LAST_BOT_QUESTION = "state:last_bot_question:{phone}"
    LAST_SEARCH = "state:last_search:{phone}"
    LAST_PRODUCT_LIST = "state:last_product_list:{phone}"
    LAST_SINGLE_PRODUCT = "state:last_single_product:{phone}"
    PENDING_QUESTION = "state:pending_question:{phone}"

    # --- Order Verification Keys ---
    AWAITING_ORDER_VERIFICATION = "state:awaiting_order_verification:{phone}"
    ORDER_VERIFIED = "state:order_verified:{phone}:{order_name}"

    # --- Triage State Keys (Refactored) ---
    TRIAGE_STATE = "state:triage:{phone}"


class TriageStates:
    """Defines the possible states in the automated triage flow."""
    AWAITING_ORDER_CONFIRM = "awaiting_order_confirm"
    AWAITING_ORDER_NUMBER = "awaiting_order_number"
    AWAITING_ISSUE_SELECTION = "awaiting_issue_selection"
    AWAITING_PHOTO = "awaiting_photo"

class TriageButtons:
    """Defines the IDs for interactive triage buttons."""
    CONFIRM_YES = "triage_confirm_yes"
    CONFIRM_NO = "triage_confirm_no"
    SELECT_ORDER_PREFIX = "triage_select_order_"
    ISSUE_DAMAGED = "triage_issue_damaged"