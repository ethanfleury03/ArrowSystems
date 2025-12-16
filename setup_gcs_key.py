#!/usr/bin/env python3
"""
GCS Key File Setup Script
Prompts user to paste base64-encoded GCS service account key and creates the file.
"""

import base64
import json
import os
import sys

def main():
    print("=" * 50)
    print("GCS Key File Setup")
    print("=" * 50)
    print()
    print("Please paste your base64-encoded GCS key JSON:")
    print("(Paste the entire base64 string, then press Enter twice to finish)")
    print()
    
    # Read base64 input (allows multi-line paste)
    lines = []
    try:
        while True:
            line = input()
            if not line.strip() and lines:  # Empty line after content means done
                break
            lines.append(line)
    except EOFError:
        pass
    
    if not lines:
        print("❌ Error: No input received")
        sys.exit(1)
    
    # Join all lines and strip whitespace
    base64_input = ''.join(lines).strip()
    
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

