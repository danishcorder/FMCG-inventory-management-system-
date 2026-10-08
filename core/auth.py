from config import USERS


def authenticate(username: str, password: str):
    user = USERS.get(username)
    if not user or user["password"] != password:
        return None
    return {"username": username, "modules": user["modules"]}
