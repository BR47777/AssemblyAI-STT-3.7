import os
import threading

_api_key_cache: tuple[tuple[str, ...], str] | None = None
_api_key_lock = threading.Lock()


def get_assemblyai_api_key() -> str | None:
    global _api_key_cache

    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if api_key:
        return api_key

    client_id = os.getenv("INFISICAL_CLIENT_ID")
    client_secret = os.getenv("INFISICAL_CLIENT_SECRET")
    project_id = os.getenv("INFISICAL_PROJECT_ID")
    if not client_id and not client_secret and not project_id:
        return None
    if not client_id or not client_secret or not project_id:
        raise RuntimeError(
            "Infisical requires INFISICAL_CLIENT_ID, "
            "INFISICAL_CLIENT_SECRET, and INFISICAL_PROJECT_ID."
        )

    environment_slug = os.getenv("INFISICAL_ENVIRONMENT_SLUG", "dev")
    secret_path = os.getenv("INFISICAL_SECRET_PATH", "/")
    secret_name = os.getenv("INFISICAL_SECRET_NAME", "ASSEMBLYAI_API_KEY")
    host = os.getenv("INFISICAL_HOST", "https://app.infisical.com")
    cache_key = (
        client_id,
        client_secret,
        project_id,
        environment_slug,
        secret_path,
        secret_name,
        host,
    )

    with _api_key_lock:
        if _api_key_cache and _api_key_cache[0] == cache_key:
            return _api_key_cache[1]

        try:
            from infisical_sdk import InfisicalSDKClient

            client = InfisicalSDKClient(host=host)
            client.auth.universal_auth.login(
                client_id=client_id,
                client_secret=client_secret,
            )
            secret = client.secrets.get_secret_by_name(
                secret_name=secret_name,
                project_id=project_id,
                environment_slug=environment_slug,
                secret_path=secret_path,
            )
            api_key = secret.secretValue
        except Exception as error:
            raise RuntimeError(
                "Unable to retrieve the AssemblyAI API key from Infisical."
            ) from error

        if not isinstance(api_key, str) or not api_key:
            raise RuntimeError(
                f"Infisical secret {secret_name!r} is empty or invalid."
            )

        _api_key_cache = (cache_key, api_key)
        return api_key
