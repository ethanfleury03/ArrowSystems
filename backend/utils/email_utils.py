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
    
    Args:
        to_email: Recipient email address
        invite_link: Full URL to the invite acceptance page
    """
    from_email = os.getenv("INVITE_FROM_EMAIL", "ethan@arrsys.com")
    from_name = os.getenv("INVITE_FROM_NAME", "Arrow Systems Support")
    subject = os.getenv("INVITE_SUBJECT", "You've been invited to Arrow Systems Support")

    body = (
        f"Hello,\n\n"
        f"You've been invited to access Arrow Systems Support.\n\n"
        f"To set your password and activate your account, click the link below:\n"
        f"{invite_link}\n\n"
        f"This link will expire in 7 days.\n\n"
        f"If you did not expect this email, you can ignore it.\n\n"
        f"Best regards,\n"
        f"{from_name}"
    )

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = os.getenv("SMTP_PORT")
    smtp_user = os.getenv("SMTP_USERNAME")
    smtp_pass = os.getenv("SMTP_PASSWORD")
    smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

    if not smtp_host or not smtp_port:
        logger.warning(
            "SMTP not configured; invite link for %s: %s", to_email, invite_link
        )
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{from_name} <{from_email}>"
    msg["To"] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
            if smtp_use_tls:
                server.starttls()
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info("Sent invite email to %s", to_email)
    except Exception as e:
        logger.exception("Failed to send invite email to %s", to_email)
        logger.warning("Invite link for %s: %s", to_email, invite_link)

