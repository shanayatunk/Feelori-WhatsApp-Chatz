#!/bin/bash
set -euo pipefail

# === CONFIG ===
# Replace these with your real tokens
PROD_APP_ACCESS_TOKEN="YOUR_PROD_APP_ACCESS_TOKEN"
DEV_APP_ACCESS_TOKEN="YOUR_DEV_APP_ACCESS_TOKEN"

# WABA IDs
PROD_WABA_ID="1122316606355933"   # Feelori Jewelry Store
TEST_WABA_ID="1270434944729993"   # Test WhatsApp Business Account

echo "🔄 Cleaning up WhatsApp App <-> WABA subscriptions..."

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

# --- Step 3: Verify PROD WABA subscriptions ---
echo "✅ Verifying PROD WABA subscriptions..."
curl -s -X GET \
  "https://graph.facebook.com/v21.0/$PROD_WABA_ID/subscribed_apps" \
  -H "Authorization: Bearer $PROD_APP_ACCESS_TOKEN" \
  | jq .

# --- Step 4: Verify TEST WABA subscriptions ---
echo "✅ Verifying TEST WABA subscriptions..."
curl -s -X GET \
  "https://graph.facebook.com/v21.0/$TEST_WABA_ID/subscribed_apps" \
  -H "Authorization: Bearer $DEV_APP_ACCESS_TOKEN" \
  | jq .

echo "🎉 Cleanup complete. Subscriptions are now isolated."
