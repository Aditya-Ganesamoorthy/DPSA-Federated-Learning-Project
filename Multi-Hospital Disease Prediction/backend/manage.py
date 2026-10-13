"""Admin CLI.   python -m backend.manage create-hospital "St Mary's Chennai"

Prints the credentials a hospital needs. Uses the SAME SECRET_KEY as the server (set it in the environment),
so a hospital created here can log in to the deployed server.
"""
from __future__ import annotations

import sys

from backend.services.auth import Auth, slugify
from backend.settings import Settings


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "create-hospital":
        print(__doc__)
        return 2
    s = Settings()
    if s.ephemeral_secret:
        print("ERROR: set SECRET_KEY in the environment (it must match the server's).", file=sys.stderr)
        return 1
    hid = slugify(argv[1])
    print(f"hospital_id / username: {hid}\npassword: {Auth(s).hospital_password(hid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
