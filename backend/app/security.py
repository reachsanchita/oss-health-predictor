"""GitHub webhook signature verification (X-Hub-Signature-256)."""
import hmac
import hashlib


def verify_signature(secret: str, body: bytes, header: str) -> bool:
    if not secret or not header or not header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)
