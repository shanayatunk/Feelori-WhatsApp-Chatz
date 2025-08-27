# /app/utils/request_utils.py
from fastapi import Request

def get_remote_address(request: Request) -> str:
    """
    Safely returns the client's IP address from a request object.
    """
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"