import logging

from google.oauth2 import service_account as sa


def setup_log(name, usage=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger



def get_credentials(service_path, scopes: list):
    """
    for a complete list of scope, please visit
    https://developers.google.com/identity/protocols/oauth2/scopes
    """
    credentials = sa.Credentials.from_service_account_file(service_path)
    scoped_credentials = credentials.with_scopes(scopes)

    return scoped_credentials
