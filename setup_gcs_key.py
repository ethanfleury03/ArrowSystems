#!/usr/bin/env python3
"""
GCS Key File Setup Script
Creates GCS service account key file from embedded base64 string.
"""

import base64
import json
import os
import sys

# Base64-encoded GCS service account key
BASE64_KEY = """ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgInByb2plY3RfaWQiOiAiYXJyb3ctcmFnLXN1cHBvcnQtcHJvZCIsCiAgInByaXZhdGVfa2V5X2lkIjogIjJlOWQ0OGFmMmRlOTdhNTc4NTVjMzhiOThiZmI3YjVjZGE5ODQwNjUiLAogICJwcml2YXRlX2tleSI6ICItLS0tLUJFR0lOIFBSSVZBVEUgS0VZLS0tLS1cbk1JSUV2Z0lCQURBTkJna3Foa2lHOXcwQkFRRUZBQVNDQktnd2dnU2tBZ0VBQW9JQkFRRHd1TTFYSGpnZG9HQ1pcbnBQUEZwVEcxbi92VE45T3NyMU1xVUgvVkhHcFh1cGpoYk0zNnRHZTNLUzFpNTBRWU1rRm1tdFdiUmtrb3B3VHlcbjBFZUdDdDdGbURPdDBiSGptTjM1RG9zZUNUd1UyYjJxd3FucUFvV1VxaU02TGkxWnZlTHNBT1VNM0dMQ1NGZHZcbmF0Zk9POGNRUTRKTFpldHhoY0NmdHp3MnNTOHJzeUpNZFI5NzMyR1lFeWlsbFNqczJUczU3SVdFRFN1QnErdTlcbmtHNFROMDRjSFNBdjN0NnBQWVVFSEx6TVlFTTRKMTFVQmdGZVRJVG1teEc2ZTY2L1k5QkFialFLSmo3bm1BMnVcbnFFT3o1NTAxWnhRcHhNeE91OVVQSThFTkh0c1NPVnBGUjBSSzRBbHNabHRkM0Q2QmE2VjFicUxDNnlwZHhISW9cbk9xT0VIUlRqQWdNQkFBRUNnZ0VBUVpIZElRTlN2c011UFB2QkNRKy8xQnpSY0EzUFkwVFlqdmloY1cvTmlic0NcbnBMakovS1hDY0pKUFVoYnpwZkdlZmUva0Nta2hTV2tCUUpDNlFzbFlPQk9HMUN4LzI2S0txQ290ZTgrQkpaN1BcbitxRU92bmJVWXhQYkI2Q0hhdC85M1ZJN0VmOUFDWExDR2svTW05a05scUVFN3p2Q1g4aFNtVTdFb0pNdjZhTlpcbkVYNE13UXBDUEwwTUlIM0o2L1VtcjlsdExoYVR1Ukg5RHo1NnVOeHpGS0pBWit2clF3NzVaekRNR3JaSStJU3pcbkc3UW9IT3Q2TWUwVGtlS1lRcWZVQXNZalNnamZyL1VSazRBMFZqNWF4WjFZUXRQSmRqeXU5aUx1aE43RFZMNkdcbmNCOVduRUd5VmRXaEpXd2JlUGh0bHlOY0JuaGtFSVJ0NDZLQXNvWkZxUUtCZ1FENC9lZkF0djFuRzcvcW5jZ2dcbk5YYkVmZ2lSNUhiZlR6NHFBNGVNOFpjbkxSVjJNSXgrS0ViOHB2OXR5Z3BrUE1DbTVZdTg4Tmh3SEpFaDFVblFcblBGNmpYQno0YXRXdWlpNW5ZSnhnS20zb2RtN0YzV2dVZmJvR2taaHdjQzg2SURPazBQV3h1RXNtYWNzenljbWVcbmRiMW5xY3RKWVpJVEJ3Wk5DR24zTlIwWk93S0JnUUQzZjA3dlpqTDl4dDdPYVE5Q2h0T1lISTBaQ2FHemViSzVcbnhZT0RVV0NKejN1eFZsUUxoYWY1elZrSVgreGZUYzZBR3BUcUpVaU9XUjRabFhIYXcreVg0cFpQY2hVVGd2QXRcbmZDSUV3ZklZTmx4QkJEaXd3ck5HWnI2SmU3WlFXZnh0aU5RczlFZDdmL0ZjN0toaWhnekFkVkRqaEpPa240RzNcbklhalE1Qkg0ZVFLQmdRREpSK0E0THZWc2k4YzZMM0NmY1BqZzRRMm1lcTBKMnJKakhsVk92K0kwSGtMeU4wN3JcbmtUM0pjbXFjWXk3QlYwZFo0OGh4NWJUaDlJWHlkc3hqMU5tZ2I0OVAxbldFMGxtdTRpK25vY1VWbmQvd2ZncHNcblBqTEtxcG96a3N6cXpTdFNla1dUOUdwejFDUWJhbTFkZHNxMWFWSlhzTk40SkQ2WXVVdUlOdHllRHdLQmdIcGRcblZSZUR2ZDQrdnRYcWw0TGF5aTdBbnZvc0N1aURXTndFVFZ0VmxwZE1IK216dHVYamRRTktoYVJnV0t0ZCtxVFhcbmY1eXNSazBxdm5rRHJFRHU4VUMrNUhYdS80Q2dFa05LeGo3MzdNd1B4RmpZejNhRkxNRzM5cWhlbytyaU1xMnRcbklzbi9GSUI0NTBwOEwxeFd6bU14SFliL2UwZS9IUkQvOFVrbUdjUnhBb0dCQU4rTnZ5NjM4UEJYSlFXUUMvaW9cbkZCQmtJeEJibFlEbG9ZOC9lTHNPRDZVcWVWQzFQeVFFc2F0TENSZlEyL2N0djNFZjlDWGpDd0ZrallZZXJKeG5cbnVuNS91Wkw0dDM2NnRhWkswMnFMU2RndGcyVWhlSWl3dmY0WWgydWN0Ym1NdkpDMW9HNnVWSjhpQTR6OWcxdEVcbkhiZXh5M1ViSFhyNkQrQ2JXdmRPMVhYL1xuLS0tLS1FTkQgUFJJVkFURSBLRVktLS0tLVxuIiwKICAiY2xpZW50X2VtYWlsIjogImNsb3VkcnVuLWRlcGxveWVyQGFycm93LXJhZy1zdXBwb3J0LXByb2QuaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLAogICJjbGllbnRfaWQiOiAiMTE0MjA3NDU3NDM0MTg2NTU1MzcyIiwKICAiYXV0aF91cmkiOiAiaHR0cHM6Ly9hY2NvdW50cy5nb29nbGUuY29tL28vb2F1dGgyL2F1dGgiLAogICJ0b2tlbl91cmkiOiAiaHR0cHM6Ly9vYXV0aDIuZ29vZ2xlYXBpcy5jb20vdG9rZW4iLAogICJhdXRoX3Byb3ZpZGVyX3g1MDlfY2VydF91cmwiOiAiaHR0cHM6Ly93d3cuZ29vZ2xlYXBpcy5jb20vb2F1dGgyL3YxL2NlcnRzIiwKICAiY2xpZW50X3g1MDlfY2VydF91cmwiOiAiaHR0cHM6Ly93d3cuZ29vZ2xlYXBpcy5jb20vcm9ib3QvdjEvbWV0YWRhdGEveDUwOS9jbG91ZHJ1bi1kZXBsb3llciU0MGFycm93LXJhZy1zdXBwb3J0LXByb2QuaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLAogICJ1bml2ZXJzZV9kb21haW4iOiAiZ29vZ2xlYXBpcy5jb20iCn0K"""

def main():
    print("=" * 50)
    print("GCS Key File Setup")
    print("=" * 50)
    print()
    
    # Use the embedded base64 string
    base64_input = BASE64_KEY.strip()
    
    # Decode base64
    try:
        json_content = base64.b64decode(base64_input).decode('utf-8')
    except Exception as e:
        print(f"❌ Error: Failed to decode base64: {e}")
        sys.exit(1)
    
    # Verify it's valid JSON
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON: {e}")
        sys.exit(1)
    
    # Write to file
    key_file_path = "/workspace/gcs-key.json"
    try:
        with open(key_file_path, 'w') as f:
            f.write(json_content)
        
        # Set permissions (read/write for owner only)
        os.chmod(key_file_path, 0o600)
    except Exception as e:
        print(f"❌ Error: Failed to write file: {e}")
        sys.exit(1)
    
    # Verify the key is complete
    pk = data.get('private_key', '')
    has_begin = 'BEGIN PRIVATE KEY' in pk
    has_end = 'END PRIVATE KEY' in pk
    pk_length = len(pk)
    
    print()
    print("✅ File created successfully!")
    print(f"   Location: {key_file_path}")
    print(f"   Has BEGIN: {has_begin}")
    print(f"   Has END: {has_end}")
    print(f"   Private key length: {pk_length} chars")
    print(f"   Client email: {data.get('client_email', 'N/A')}")
    print(f"   Project ID: {data.get('project_id', 'N/A')}")
    
    if not has_end:
        print()
        print("⚠️  WARNING: Private key appears incomplete!")
        print("   The key should end with '-----END PRIVATE KEY-----'")
        sys.exit(1)
    
    if pk_length < 1500:
        print()
        print("⚠️  WARNING: Private key seems too short!")
        print("   Expected length: ~1600-1700 chars")
        print("   Your length: {} chars".format(pk_length))
    
    print()
    print("=" * 50)
    print("✅ All checks passed!")
    print("=" * 50)
    print()
    print("Next steps:")
    print("  export GOOGLE_APPLICATION_CREDENTIALS=\"/workspace/gcs-key.json\"")
    print("  python -c \"from google.cloud import storage; client = storage.Client(); print('✅ GCS access works!')\"")

if __name__ == "__main__":
    main()

