import reflex as rx


class KakaoSyncConfig(rx.Config):
    pass


config = KakaoSyncConfig(
    app_name="kakaosync",
    db_url="sqlite:///kakaosync.db",
    env=rx.Env.DEV,
    frontend_packages=[
        "react-loading-icons",
    ],
)
