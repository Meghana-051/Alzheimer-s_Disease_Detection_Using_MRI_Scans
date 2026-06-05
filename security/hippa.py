# security/hipaa.py
from __future__ import annotations
import hashlib, os, json, datetime
from cryptography.fernet import Fernet
from fastapi import Request, HTTPException
from jose import JWTError, jwt

class PHIEncryptor:
    def __init__(self, key: bytes | None = None):
        self._fernet = Fernet(key or Fernet.generate_key())

    def encrypt(self, plaintext: bytes) -> bytes:
        return self._fernet.encrypt(plaintext)

    def decrypt(self, token: bytes) -> bytes:
        return self._fernet.decrypt(token)

    def encrypt_dicom(self, raw_dicom: bytes, output_path: str) -> str:
        cipher = self.encrypt(raw_dicom)
        with open(output_path, "wb") as f:
            f.write(cipher)
        return hashlib.sha256(raw_dicom).hexdigest()

def deidentify_record(record: dict) -> dict:
    cleaned = {}
    for k, v in record.items():
        if k.lower() in {"name", "dob", "address", "phone", "email", "ssn"}:
            cleaned[k] = hashlib.sha256(str(v).encode()).hexdigest()[:12]
        elif k.lower() == "age" and isinstance(v, (int, float)) and v > 89:
            cleaned[k] = 90
        else:
            cleaned[k] = v
    return cleaned

class AuditLogger:
    def __init__(self, log_path: str = "audit/hipaa_audit.jsonl"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._path = log_path

    def log(self, event: str, actor: str, resource: str, outcome: str, meta: dict = {}):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "event": event, "actor": actor, "resource": resource, "outcome": outcome, **meta
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")

# ── FIXED: ENVIRONMENT SAFE FALLBACK KEY MAP INTEGRATION ──────────────────
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "DEVELOPMENT_STAGE_FALLBACK_KEY_HEX_123456")
ALGORITHM  = "HS256"

def require_clinician(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token.")
    token = auth.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if "clinician" not in payload.get("roles", []):
            raise HTTPException(403, "Insufficient role validation boundaries.")
        return payload
    except JWTError as e:
        raise HTTPException(401, f"Invalid token sequence mapping: {e}")