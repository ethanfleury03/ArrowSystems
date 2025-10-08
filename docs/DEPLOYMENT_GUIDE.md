# DuraFlex Technical Assistant - Deployment Guide

Complete guide for deploying the DuraFlex Technical Assistant UI application.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Production Deployment](#production-deployment)
4. [Cloud Deployment Options](#cloud-deployment-options)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **RAM**: Minimum 8GB, 16GB recommended
- **Disk Space**: 10GB minimum (for models and data)
- **GPU**: Optional but recommended for faster inference

### Required Software
- Python 3.9+
- pip (Python package manager)
- Git
- Virtual environment tool (venv or conda)

---

## Local Development Setup

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd rag_app.py
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Configuration

#### Create User Accounts
Edit `config/users.yaml` to add your technicians:

```python
# Generate password hash
import hashlib
password = "your_password"
salt = "arrow_secure_2024"
hash = hashlib.sha256((password + salt).encode()).hexdigest()
print(hash)
```

Add to `config/users.yaml`:
```yaml
credentials:
  usernames:
    your_username:
      email: user@company.com
      name: Full Name
      password: <generated_hash>
      salt: arrow_secure_2024
      role: technician  # or admin
```

#### Configure Application
Edit `config/app_config.yaml` to customize settings.

### 5. Run the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

---

## Production Deployment

### Option 1: Docker Deployment (Recommended)

#### Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run
```bash
# Build image
docker build -t duraflex-assistant .

# Run container
docker run -p 8501:8501 \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/data:/app/data \
  duraflex-assistant
```

### Option 2: Docker Compose (For full stack)

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./storage:/app/storage
      - ./data:/app/data
      - ./extracted_content:/app/extracted_content
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
```

Run:
```bash
docker-compose up -d
```

---

## Cloud Deployment Options

### 1. Streamlit Cloud (Easiest)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy DuraFlex Assistant"
   git push
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `app.py` as main file
   - Deploy!

3. **Configure Secrets**
   - In Streamlit Cloud dashboard, go to Settings → Secrets
   - Copy contents from `.streamlit/secrets.toml`

**Note**: Free tier has limitations. Consider paid plan for production.

### 2. AWS EC2 Deployment

#### Launch EC2 Instance
- Instance type: t3.large or better (for GPU: g4dn.xlarge)
- OS: Ubuntu 22.04 LTS
- Storage: 30GB minimum
- Security group: Allow port 8501

#### Setup Script
```bash
#!/bin/bash

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install -y python3.10 python3-pip python3-venv git

# Clone repository
git clone <your-repo-url>
cd rag_app.py

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install systemd service
sudo tee /etc/systemd/system/duraflex.service > /dev/null <<EOF
[Unit]
Description=DuraFlex Technical Assistant
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/rag_app.py
Environment="PATH=/home/ubuntu/rag_app.py/venv/bin"
ExecStart=/home/ubuntu/rag_app.py/venv/bin/streamlit run app.py --server.port=8501
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl daemon-reload
sudo systemctl enable duraflex
sudo systemctl start duraflex
```

#### Setup Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### 3. Azure App Service

```bash
# Install Azure CLI
# ...

# Login
az login

# Create resource group
az group create --name duraflex-rg --location eastus

# Create App Service plan
az appservice plan create \
  --name duraflex-plan \
  --resource-group duraflex-rg \
  --sku B2 \
  --is-linux

# Create web app
az webapp create \
  --resource-group duraflex-rg \
  --plan duraflex-plan \
  --name duraflex-assistant \
  --runtime "PYTHON|3.10"

# Deploy
az webapp up --name duraflex-assistant
```

### 4. Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/duraflex-assistant

# Deploy to Cloud Run
gcloud run deploy duraflex-assistant \
  --image gcr.io/PROJECT-ID/duraflex-assistant \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## Configuration

### Environment Variables

Create `.env` file:
```env
# Application
APP_ENV=production
LOG_LEVEL=INFO

# Paths
STORAGE_DIR=/app/storage
DATA_DIR=/app/data
CACHE_DIR=/root/.cache/huggingface/hub

# Session
SESSION_TIMEOUT_HOURS=24

# Security
SECRET_KEY=your-secret-key-here
```

### SSL/HTTPS Setup

#### Using Let's Encrypt (Free)
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### Firewall Configuration
```bash
# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8501/tcp

# Enable firewall
sudo ufw enable
```

---

## User Management

### Adding New Users

1. **Generate Password Hash**:
```python
import hashlib
password = "user_password"
salt = "arrow_secure_2024"
hash_value = hashlib.sha256((password + salt).encode()).hexdigest()
print(f"Password hash: {hash_value}")
```

2. **Add to config/users.yaml**:
```yaml
new_user:
  email: newuser@company.com
  name: New User
  password: <generated_hash>
  salt: arrow_secure_2024
  role: technician
```

3. **Restart Application**:
```bash
sudo systemctl restart duraflex
```

---

## Monitoring and Maintenance

### Log Files
- Application logs: `rag_handler.log`
- Streamlit logs: Check systemd journal
  ```bash
  sudo journalctl -u duraflex -f
  ```

### Backup Strategy
```bash
# Backup script
#!/bin/bash
BACKUP_DIR=/backups/duraflex
DATE=$(date +%Y%m%d_%H%M%S)

# Backup storage
tar -czf $BACKUP_DIR/storage_$DATE.tar.gz storage/

# Backup config
tar -czf $BACKUP_DIR/config_$DATE.tar.gz config/

# Backup user data
tar -czf $BACKUP_DIR/extracted_content_$DATE.tar.gz extracted_content/

# Keep only last 7 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

### Performance Monitoring
- Use `htop` for resource monitoring
- Monitor GPU with `nvidia-smi` (if applicable)
- Track query latency in logs

---

## Troubleshooting

### Common Issues

#### 1. Models not loading
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface/hub
# Restart application
```

#### 2. Port already in use
```bash
# Kill process on port 8501
sudo lsof -ti:8501 | xargs kill -9
```

#### 3. Out of memory
- Reduce `top_k` in config
- Use smaller model variant
- Increase system RAM
- Enable swap space

#### 4. Slow performance
- Enable GPU acceleration
- Reduce batch size
- Use caching
- Optimize chunk size

### Getting Help
- Check logs: `rag_handler.log`
- Review Streamlit documentation
- Contact Arrow Systems support

---

## Security Best Practices

1. **Change Default Passwords**: Update all default credentials
2. **Use HTTPS**: Always use SSL/TLS in production
3. **Update Regularly**: Keep dependencies updated
4. **Backup Regularly**: Automated backup strategy
5. **Monitor Access**: Review authentication logs
6. **Firewall**: Restrict access to necessary ports only
7. **Secrets Management**: Never commit secrets to version control

---

## Scaling

### Horizontal Scaling
- Use load balancer (nginx, HAProxy)
- Deploy multiple instances
- Share storage via NFS or S3

### Vertical Scaling
- Upgrade instance size
- Add GPUs for faster inference
- Increase RAM for larger indices

---

## License & Support

© 2025 Arrow Systems Inc

For enterprise support, contact: support@arrowsystems.com

