#!/bin/bash
set -euo pipefail

# === CONFIG ===
# Replace with your real tokens
# Tokens must come from environment
: "${PROD_APP_ACCESS_TOKEN:?Set PROD_APP_ACCESS_TOKEN in env}"
: "${DEV_APP_ACCESS_TOKEN:?Set DEV_APP_ACCESS_TOKEN in env}"

# App IDs
PROD_APP_ID="1062440482164641"    # Feelori Chatz (Prod)
DEV_APP_ID="1948972509285197"     # feelori-dev (Test)

# WABA IDs
PROD_WABA_ID="1122316606355933"   # Feelori Jewelry Store
TEST_WABA_ID="1270434944729993"   # Test WhatsApp Business Account

# Expected webhook URLs
PROD_WEBHOOK_URL="https://api.feelori.com/api/v1/webhooks/whatsapp"
TEST_WEBHOOK_URL="https://staging-api.feelori.com/api/v1/webhooks/whatsapp"

echo "🔄 Resetting WhatsApp App <-> WABA subscriptions..."

# --- Step 1: Unsubscribe PROD app from TEST WABA ---
echo "👉 Unsubscribing PROD app from TEST WABA..."
curl -s -X DELETE \
  "https://graph.facebook.com/v21.0/$TEST_WABA_ID/subscribed_apps" \
  -H "Authorization: Bearer $PROD_APP_ACCESS_TOKEN" \
  | jq .

# --- Step 2: Unsubscribe DEV app from PROD WABA ---
echo "👉 Unsubscribing DEV app from PROD WABA..."
curl -s -X DELETE \
  "https://graph.facebook.com/v21.0/$PROD_WABA_ID/subscribed_apps" \
  -H "Authorization: Bearer $DEV_APP_ACCESS_TOKEN" \
  | jq .

# --- Step 3: Re-subscribe PROD app to PROD WABA ---
echo "🔗 Re-subscribing PROD app ($PROD_APP_ID) to PROD WABA..."
curl -s -X POST \
  "https://graph.facebook.com/v21.0/$PROD_WABA_ID/subscribed_apps" \
  -H "Authorization: Bearer $PROD_APP_ACCESS_TOKEN" \
  | jq .

# --- Step 4: Re-subscribe DEV app to TEST WABA ---
echo "🔗 Re-subscribing DEV app ($DEV_APP_ID) to TEST WABA..."
curl -s -X POST \
  "https://graph.facebook.com/v21.0/$TEST_WABA_ID/subscribed_apps" \
  -H "Authorization: Bearer $DEV_APP_ACCESS_TOKEN" \
  | jq .

# --- Step 5: Verify PROD WABA subscriptions ---
echo "✅ Verifying PROD WABA subscriptions..."
curl -s -X GET \
  "https://graph.facebook.com/v21.0/$PROD_WABA_ID/subscribed_apps" \
  -H "Authorization: Bearer $PROD_APP_ACCESS_TOKEN" \
  | jq .

# --- Step 6: Verify TEST WABA subscriptions ---
echo "✅ Verifying TEST WABA subscriptions..."
curl -s -X GET \
  "https://graph.facebook.com/v21.0/$TEST_WABA_ID/subscribed_apps" \
  -H "Authorization: Bearer $DEV_APP_ACCESS_TOKEN" \
  | jq .

# --- Step 7: Check Webhook URLs ---
echo "🔍 Checking webhook URLs..."

PROD_WEBHOOK=$(curl -s -X GET \
  "https://graph.facebook.com/v21.0/$PROD_APP_ID/subscriptions" \
  -H "Authorization: Bearer $PROD_APP_ACCESS_TOKEN" \
  | jq -r '.data[0].callback_url // empty')

TEST_WEBHOOK=$(curl -s -X GET \
  "https://graph.facebook.com/v21.0/$DEV_APP_ID/subscriptions" \
  -H "Authorization: Bearer $DEV_APP_ACCESS_TOKEN" \
  | jq -r '.data[0].callback_url // empty')

echo "Prod webhook: $PROD_WEBHOOK"
echo "Test webhook: $TEST_WEBHOOK"

if [[ "$PROD_WEBHOOK" == "$PROD_WEBHOOK_URL" ]]; then
  echo "✅ PROD webhook URL is correct."
else
  echo "❌ PROD webhook mismatch! Expected: $PROD_WEBHOOK_URL"
fi

if [[ "$TEST_WEBHOOK" == "$TEST_WEBHOOK_URL" ]]; then
  echo "✅ TEST webhook URL is correct."
else
  echo "❌ TEST webhook mismatch! Expected: $TEST_WEBHOOK_URL"
fi

echo "🎉 Reset complete. Correct apps + webhooks are verified."
