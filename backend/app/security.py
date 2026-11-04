import os
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext

SECRET = os.getenv("JWT_SECRET", "development-only-change-this-secret-before-deployment")
ALGORITHM = "HS256"
pwd = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated=["bcrypt"])

def hash_password(password: str) -> str:
    return pwd.hash(password)

def verify_password(password: str, password_hash: str) -> bool:
    return pwd.verify(password, password_hash)

def create_token(user_id: int) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=int(os.getenv("JWT_EXPIRE_MINUTES", "1440")))
    return jwt.encode({"sub": str(user_id), "exp": expires}, SECRET, algorithm=ALGORITHM)

def token_user_id(token: str) -> int | None:
    try:
        return int(jwt.decode(token, SECRET, algorithms=[ALGORITHM])["sub"])
    except (JWTError, KeyError, TypeError, ValueError):
        return None
