"""Email utilities for sending invite emails."""

import os
import smtplib
import logging
from email.message import EmailMessage

logger = logging.getLogger(__name__)


def send_invite_email(to_email: str, invite_link: str) -> None:
    """
    Send an invite email with the given link.
    
    If SMTP config is missing, logs the link so an admin can manually send it.
    If SMTP is configured but sending fails, logs the exception and falls back to log-only.
    
    Args:
        to_email: Recipient email address
        invite_link: Full URL to the invite acceptance page
    """
    # Read invite email settings with safe defaults
    from_email = os.getenv("INVITE_FROM_EMAIL", "ethan@arrsys.com")
    from_name = os.getenv("INVITE_FROM_NAME", "Arrow Systems Support")
    subject = os.getenv("INVITE_SUBJECT", "You've been invited to Arrow Systems Support")

    # Read SMTP configuration
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = os.getenv("SMTP_PORT")
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes", "on")

    # Check if SMTP is configured (host and port are required)
    smtp_configured = bool(smtp_host and smtp_port)

    if not smtp_configured:
        logger.warning(
            "SMTP not configured; invite link for %s: %s", to_email, invite_link
        )
        return

    # Build email message
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = f"{from_name} <{from_email}>"
    message["To"] = to_email
    message.set_content(
        f"You've been invited to Arrow Systems Support.\n\n"
        f"Click this link to set your password and activate your account:\n\n"
        f"{invite_link}\n\n"
        f"If you did not expect this email, you can ignore it."
    )

    # Attempt to send via SMTP
    try:
        port = int(smtp_port)
        with smtplib.SMTP(smtp_host, port, timeout=30) as server:
            if smtp_use_tls:
                server.starttls()
            if smtp_username and smtp_password:
                server.login(smtp_username, smtp_password)
            server.send_message(message)
        logger.info("Invite email successfully sent to %s", to_email)
    except Exception:
        logger.exception(
            "Failed to send invite email via SMTP to %s; falling back to log-only invite link: %s",
            to_email,
            invite_link,
        )
        # Fallback: still log the link so we can copy-paste in staging
        logger.warning("Invite link for %s: %s", to_email, invite_link)

