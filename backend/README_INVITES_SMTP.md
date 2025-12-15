# Invite Emails and SMTP Configuration

The backend sends invite emails via the `send_invite_email` helper in `backend/utils/email_utils.py`.

## Environment Variables

The following env vars control invite links and SMTP:

- `FRONTEND_BASE_URL`  
  Base URL used in invite links. Example:  
  `https://arrow-rag-frontend-70705019874.us-central1.run.app` (default)  
  Later you can set this to your custom domain, e.g. `https://app.arrsys.com`.

- `INVITE_FROM_EMAIL` (default: `ethan@arrsys.com`)  
- `INVITE_FROM_NAME` (default: `Arrow Systems Support`)  
- `INVITE_SUBJECT` (default: `You've been invited to Arrow Systems Support`)

- `SMTP_HOST` (required to actually send email)  
- `SMTP_PORT` (e.g., `587`)  
- `SMTP_USERNAME`  
- `SMTP_PASSWORD`  
- `SMTP_USE_TLS` (`true` or `false`, default `true`)

If `SMTP_HOST` or `SMTP_PORT` is missing, the code will **not send email** and will log:

> `SMTP not configured; invite link for <email>: <link>`

This is useful in local/staging environments where you just want the link from logs.

## Example: Configure SMTP via gcloud (Gmail / Google Workspace)

For early testing you can use a Gmail or Workspace account with an **App Password**:

```bash
gcloud run services update arrow-rag-backend \
  --region=us-central1 \
  --set-env-vars=SMTP_HOST=smtp.gmail.com \
  --set-env-vars=SMTP_PORT=587 \
  --set-env-vars=SMTP_USERNAME=ethan@arrsys.com \
  --set-env-vars=SMTP_PASSWORD=YOUR_APP_PASSWORD_HERE \
  --set-env-vars=SMTP_USE_TLS=true \
  --set-env-vars=INVITE_FROM_EMAIL=ethan@arrsys.com \
  --set-env-vars=INVITE_FROM_NAME="Arrow Systems Support" \
  --set-env-vars=INVITE_SUBJECT="You've been invited to Arrow Systems Support" \
  --set-env-vars=FRONTEND_BASE_URL=https://arrow-rag-frontend-70705019874.us-central1.run.app
```

Replace `YOUR_APP_PASSWORD_HERE` with the 16-character app password from Google.

After setting these, new users created in the Admin → Users page will receive an invite email with a link to `/accept-invite?token=...`, where they can set their password and log in.

