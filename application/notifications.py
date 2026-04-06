from __future__ import annotations

import traceback
from dataclasses import dataclass

import streamlit as st


@dataclass
class NotificationConfig:
    """Configuration for notification appearance."""

    toast_duration_ms: int = 4000
    show_details_in_expander: bool = True


_config = NotificationConfig()


def configure(
    toast_duration_ms: int | None = None,
    show_details_in_expander: bool | None = None,
) -> None:
    """Configure global notification settings."""
    if toast_duration_ms is not None:
        _config.toast_duration_ms = toast_duration_ms
    if show_details_in_expander is not None:
        _config.show_details_in_expander = show_details_in_expander


def _format_exception(exc: BaseException) -> str:
    """Format exception with traceback for details view."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def show_error(
    message: str,
    *,
    details: str | None = None,
    exception: BaseException | None = None,
    use_toast: bool = True,
) -> None:
    """Show an error notification.

    Args:
        message: Short, user-friendly error message
        details: Optional detailed error information
        exception: Optional exception object (will format traceback)
        use_toast: If True, show as toast; if False, show inline
    """
    if exception is not None:
        details = _format_exception(exception)

    if use_toast:
        st.toast(message, icon=":material/error:")

    if details and _config.show_details_in_expander:
        with st.expander(f"Error: {message}", expanded=False):
            st.code(details, language="text")
    elif not use_toast:
        st.error(message)


def show_warning(
    message: str,
    *,
    details: str | None = None,
    use_toast: bool = True,
) -> None:
    """Show a warning notification.

    Args:
        message: Short warning message
        details: Optional detailed information
        use_toast: If True, show as toast; if False, show inline
    """
    if use_toast:
        st.toast(message, icon=":material/warning:")

    if details and _config.show_details_in_expander:
        with st.expander(f"Warning: {message}", expanded=False):
            st.text(details)
    elif not use_toast:
        st.warning(message)


def show_success(
    message: str,
    *,
    use_toast: bool = True,
) -> None:
    """Show a success notification.

    Args:
        message: Success message
        use_toast: If True, show as toast; if False, show inline
    """
    if use_toast:
        st.toast(message, icon=":material/check_circle:")
    else:
        st.success(message)


def show_info(
    message: str,
    *,
    use_toast: bool = True,
) -> None:
    """Show an info notification.

    Args:
        message: Info message
        use_toast: If True, show as toast; if False, show inline
    """
    if use_toast:
        st.toast(message, icon=":material/info:")
    else:
        st.info(message)


class _NotifyNamespace:
    """Namespace for notify.error(), notify.warning(), etc."""

    @staticmethod
    def error(
        message: str,
        *,
        details: str | None = None,
        exception: BaseException | None = None,
    ) -> None:
        show_error(message, details=details, exception=exception)

    @staticmethod
    def warning(message: str, *, details: str | None = None) -> None:
        show_warning(message, details=details)

    @staticmethod
    def success(message: str) -> None:
        show_success(message)

    @staticmethod
    def info(message: str) -> None:
        show_info(message)


notify = _NotifyNamespace()


def handle_exception(
    exc: BaseException,
    message: str = "An error occurred",
    *,
    reraise: bool = False,
) -> None:
    """Handle an exception by showing a notification.

    Args:
        exc: The exception that was caught
        message: User-friendly message to display
        reraise: If True, re-raise the exception after showing notification
    """
    show_error(message, exception=exc)
    if reraise:
        raise exc
