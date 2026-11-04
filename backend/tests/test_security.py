import unittest

from fastapi import HTTPException
from passlib.context import CryptContext

from app.main import get_job
from app.security import hash_password, verify_password


class FakeDatabase:
    def __init__(self):
        self.calls = []

    def execute(self, query, parameters=()):
        self.calls.append((query, parameters))
        return self

    def fetchone(self):
        return None


class SecurityTests(unittest.TestCase):
    def test_password_hash_supports_long_passwords_and_legacy_bcrypt(self):
        long_password = "p" * 100
        self.assertTrue(verify_password(long_password, hash_password(long_password)))
        legacy_hash = CryptContext(schemes=["bcrypt"]).hash("legacy-password")
        self.assertTrue(verify_password("legacy-password", legacy_hash))

    def test_private_job_reads_are_scoped_to_the_current_user(self):
        database = FakeDatabase()
        with self.assertRaises(HTTPException) as raised:
            get_job(42, user={"id": 7}, db=database)
        self.assertEqual(raised.exception.status_code, 404)
        query, parameters = database.calls[0]
        self.assertIn("owner_id=%s", query)
        self.assertEqual(parameters, (42, 7))


if __name__ == "__main__":
    unittest.main()