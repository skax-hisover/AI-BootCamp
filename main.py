"""Entry point for final-project local execution."""

from src.config.settings import load_settings


if __name__ == "__main__":
    settings = load_settings()
    print("Final project bootstrap ready")
    print(f"endpoint={settings.aoai_endpoint}")
    print(f"deployment={settings.aoai_deployment}")
    print(f"api_version={settings.aoai_api_version}")
