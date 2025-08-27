# /app/utils/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# This file defines all Prometheus metrics used for application monitoring.
# Centralizing them here makes them easy to find and manage.

# Business Logic Metrics
message_counter = Counter('whatsapp_messages_total', 'Total messages processed', ['status', 'message_type'])
response_time_histogram = Histogram('response_time_seconds', 'Response time in seconds', ['endpoint'])
active_customers_gauge = Gauge('active_customers', 'Number of active customers')
ai_requests_counter = Counter('ai_requests_total', 'Total AI requests', ['model', 'status'])
database_operations_counter = Counter('database_operations_total', 'Database operations', ['operation', 'status'])

# Security Metrics
auth_attempts_counter = Counter('auth_attempts_total', 'Authentication attempts', ['status', 'method'])
webhook_signature_counter = Counter('webhook_signature_verifications_total', 'Webhook signature verifications', ['status'])

# Performance Metrics
cache_operations = Counter('cache_operations_total', 'Cache operations', ['operation', 'status'])