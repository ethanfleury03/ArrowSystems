# Tools Directory

This directory contains development tools and utilities.

## Cloud SQL Auth Proxy

Place the Cloud SQL Auth Proxy v2 Windows executable here:

```
tools/cloud-sql-proxy.exe
```

### Download Instructions

1. Visit: https://cloud.google.com/sql/docs/postgres/sql-proxy#install
2. Download the Windows 64-bit executable
3. Rename it to `cloud-sql-proxy.exe`
4. Place it in this directory

The proxy is used by the VS Code task "Dev: Cloud SQL Proxy" to connect to Cloud SQL instances locally on Windows.
