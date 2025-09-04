# backend/gunicorn_conf.py

# Gunicorn config file

# Basic configuration
bind = "0.0.0.0:8000"
workers = 2
worker_class = "uvicorn.workers.UvicornWorker"

# Settings for running behind a reverse proxy like Nginx
forwarded_allow_ips = "*"
proxy_protocol = True
proxy_allow_ips = '*'

# --- Logging ---
# Send access and error logs to stdout and stderr
accesslog = "-"
errorlog = "-"
# Set the log level
loglevel = "info"