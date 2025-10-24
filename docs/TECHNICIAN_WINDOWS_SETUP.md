# 🚀 DuraFlex Technical Assistant - Windows Deployment Guide

## For Technician's Laptop - Using start.sh (No .bat files needed!)

---

## ✅ **ONE-TIME SETUP (5 minutes)**

### Step 1: Open Git Bash
- Right-click on your Desktop
- Select **"Git Bash Here"**
- If you don't see this option → Install Git from: https://git-scm.com/download/win

### Step 2: Clone the Repository
```bash
git clone https://github.com/ethanfleury03/ArrowSystems.git
cd ArrowSystems
```

### Step 3: Run start.sh
```bash
./start.sh
```

**That's it!** 🎉

The `start.sh` script will:
- ✅ Automatically detect Windows environment
- ✅ Create Python virtual environment
- ✅ Install all dependencies
- ✅ Download AI models (~2GB)
- ✅ Setup AWS DynamoDB
- ✅ Start the application

⏱️ **First run:** 10-15 minutes (downloads everything)  
⏱️ **Subsequent runs:** 5-10 seconds

---

## 🌐 Access the App

Once started, open your browser to:
```
http://localhost:8501
```

### 🔐 Login Credentials:
- **Admin:** `admin` / `admin123`
- **Technician:** `tech1` / `tech123`

---

## 🛑 Stopping the App

In the Git Bash window:
- Press **`Ctrl + C`**

---

## 🔄 Starting Again Later

Just open Git Bash and run:
```bash
cd ArrowSystems
./start.sh
```

**Note:** You can also right-click on the ArrowSystems folder and select "Git Bash Here", then just run:
```bash
./start.sh
```

---

## 🐛 Troubleshooting

### "bash: python: command not found"
**Fix:** Install Python from https://www.python.org/downloads/
- During installation, check ☑️ **"Add Python to PATH"**
- After installing, close and reopen Git Bash

### "git: command not found"
**Fix:** Install Git from https://git-scm.com/download/win
- Use default settings during installation

### Port 8501 already in use
**Fix:** Stop existing Streamlit process:
```bash
taskkill //F //IM python.exe //FI "WINDOWTITLE eq *streamlit*"
```
Or just close the previous Git Bash window running the app.

### Need to update the app
```bash
cd ArrowSystems
git pull
./start.sh
```

---

## 📝 Why Git Bash?

**Git Bash** is a Windows application that provides a bash shell - it can run `.sh` scripts directly on Windows!

- ✅ No need for `.bat` files
- ✅ No need for WSL (Windows Subsystem for Linux)
- ✅ Same `start.sh` script works on Windows, Linux, and Mac
- ✅ Simpler deployment - one script for all platforms

**Git is usually already installed** on most development machines. If not, it's a quick 2-minute install.

---

## 💡 What start.sh Does Automatically

The `start.sh` script is smart and handles everything:

- ✅ **Detects Windows/Linux** automatically
- ✅ **Creates virtual environment** if needed
- ✅ **Installs Python packages** (only missing ones)
- ✅ **Downloads AI models** (~2GB, first time only)
- ✅ **Sets up AWS DynamoDB** credentials
- ✅ **Starts the application** on port 8501
- ✅ **Shows login credentials** when ready

---

## 📊 What to Expect

### First Startup (One-time):
```
🔧 DuraFlex Technical Assistant
==========================================

🖥️  Environment: Local Machine

📦 Creating virtual environment...
✅ Virtual environment created

🐍 Python version: 3.11.5

🔍 Checking dependencies...
  ❌ Streamlit not found
  ❌ PyTorch not found

📥 Installing all dependencies...
   (This may take a few minutes...)
✅ All dependencies installed

🤖 Checking Claude for LLM answer generation...
  ✅ Claude API key found
  🎉 LLM answer generation enabled!

✅ RAG index found in latest_model/
   📊 Indexed chunks: 1,247

🚀 Starting Streamlit server...

==========================================
  Local URL: http://localhost:8501
==========================================
```

### Subsequent Startups (Fast):
```
🔧 DuraFlex Technical Assistant
==========================================

✅ All dependencies satisfied!
✅ RAG index found
🚀 Starting Streamlit server...

  Local URL: http://localhost:8501
```

---

## ✅ Quick Commands Reference

| Task | Command |
|------|---------|
| **Start App** | `./start.sh` |
| **Stop App** | `Ctrl + C` |
| **Update App** | `git pull` then `./start.sh` |
| **Check Status** | Look at terminal output |
| **View Logs** | Scroll up in terminal |

---

## 🎯 Testing the Installation

1. ✅ Start the app with `./start.sh`
2. ✅ Open browser to `http://localhost:8501`
3. ✅ Login as admin: `admin` / `admin123`
4. ✅ Ask a test question: "What is DuraFlex?"
5. ✅ Verify you get a response with sources

---

## 📞 Support

If issues occur:
1. Check the terminal output for error messages
2. Screenshot any errors
3. Contact: Ethan Fleury

---

## ✨ Benefits of Using start.sh

✅ **No manual setup** - Everything is automated  
✅ **Cross-platform** - Works on Windows (Git Bash), Linux, Mac  
✅ **Smart detection** - Skips already-installed packages  
✅ **AWS DynamoDB** - Automatically configured  
✅ **Production-ready** - All features enabled  
✅ **Self-healing** - Installs missing dependencies automatically  

---

**Version:** 1.0  
**Last Updated:** October 22, 2025  
**Tested On:** Windows 10/11, Git Bash 2.42+

