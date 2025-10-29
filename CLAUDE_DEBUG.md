# Claude API Key Issue - Debugging Steps

## ✅ What We Know

1. **API Key is correct** - You confirmed it works outside Docker
2. **Model name is correct** - `claude-sonnet-4-20250514` works when tested directly
3. **Direct test works** - API call succeeds when run manually in container
4. **Container can reach API** - Network connectivity is fine

## 🔍 Why It's Failing During Init

The error happens during **background initialization** when the app starts. Possible causes:

1. **Timing issue** - Background thread might be racing
2. **Rate limiting** - Multiple simultaneous init attempts might hit limits
3. **Error handling** - Generic "Connection error" might be masking real issue

## 🛠️ Next Steps

Rebuild with improved error logging:

```powershell
# Rebuild with better error messages
.\build-local.ps1

# Run container
.\run-local.ps1
```

Check the logs - you should now see the **actual error type and message** instead of just "Connection error".

## 📊 Current Status

- ✅ Dockerfile fixed (cache directory, environment variables)
- ✅ Code updated (uses HF_HOME env var)
- ✅ Error logging improved (will show full error details)
- ⚠️ Need to rebuild to see actual error

After rebuild, check the container logs for the full error message!

