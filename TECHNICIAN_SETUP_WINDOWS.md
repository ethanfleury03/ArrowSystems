# 🚀 DuraFlex Technical Assistant - Windows Laptop Setup

## Quick Setup Guide for Technicians

### ⚙️ Prerequisites
- Windows 10/11
- Python 3.9 or newer ([Download here](https://www.python.org/downloads/) if needed)
- Internet connection (for first-time setup)

---

## 📋 Step-by-Step Installation

### Use **PowerShell** (Recommended) or **Command Prompt**

---

## 🟢 OPTION A: Automated Setup (Easiest)

### Step 1: Open PowerShell
- Press `Windows + X`
- Click "Windows PowerShell" or "Terminal"

### Step 2: Navigate to where you want to install
```powershell
cd C:\Users\YourUsername\Documents
```

### Step 3: Clone the repository
```powershell
git clone https://github.com/ethanfleury03/ArrowSystems.git
cd ArrowSystems
```

### Step 4: Run the setup script
```powershell
python setup_and_start.bat
```

**That's it!** The app will:
- ✅ Install all dependencies automatically
- ✅ Download AI models (~2GB, takes 5-10 minutes)
- ✅ Start the application
- ✅ Open in your browser automatically

**⏱️ Total time: 10-15 minutes**

---

## 🔵 OPTION B: Manual Setup (Step-by-Step)

### Step 1: Open PowerShell as Administrator
- Press `Windows + X`
- Click "Windows PowerShell (Admin)" or "Terminal (Admin)"

### Step 2: Navigate to where you want to install
```powershell
cd C:\Users\YourUsername\Documents
```

### Step 3: Clone the repository
```powershell
git clone https://github.com/ethanfleury03/ArrowSystems.git
```

### Step 4: Navigate into the folder
```powershell
cd ArrowSystems
```

### Step 5: Install Python dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
⏱️ *This takes 3-5 minutes*

### Step 6: Start the application
```powershell
python -m streamlit run app.py
```

### Step 7: Open your browser
The app should open automatically. If not, go to:
```
http://localhost:8501
```

---

## 🔐 Login Credentials

**Default Admin Account:**
- Username: `admin`
- Password: `admin123`

**Important:** Change the password after first login!

---

## 🎯 Quick Test

After the app starts:

1. ✅ **Login** with admin credentials
2. ✅ **Type a test question**: "What is DuraFlex?"
3. ✅ **Check response**: You should get a detailed answer with sources

---

## 🛑 Stopping the Application

To stop the app:
- Press `Ctrl + C` in the PowerShell window
- Or just close the PowerShell window

---

## 🔄 Starting Again (After First Setup)

Once installed, you only need:

```powershell
cd C:\Users\YourUsername\Documents\ArrowSystems
python -m streamlit run app.py
```

Or just double-click the `START_APP.bat` file!

---

## 🐛 Troubleshooting

### Issue: "Python is not recognized"
**Fix:** Install Python from https://www.python.org/downloads/
- ✅ Check "Add Python to PATH" during installation

### Issue: "git is not recognized"
**Fix:** Install Git from https://git-scm.com/download/win
- ✅ Use default settings during installation

### Issue: Port 8501 is already in use
**Fix:**
```powershell
# Stop any existing Streamlit processes
taskkill /F /IM streamlit.exe
```

### Issue: Slow first startup
**Normal!** The first time:
- Downloads AI models (~2GB)
- Takes 10-15 minutes
- Subsequent startups are much faster (5-10 seconds)

### Issue: Models downloading every time
**Fix:** The models are cached. Make sure you're running from the same folder each time.

---

## 📞 Support

If you encounter issues:
1. Check the console for error messages
2. Take a screenshot of any errors
3. Contact: Ethan Fleury

---

## 🎉 You're Ready!

The DuraFlex Technical Assistant is now running on your laptop.

**What you can do:**
- ✅ Ask technical questions about DuraFlex products
- ✅ Search documentation instantly
- ✅ Get troubleshooting guidance
- ✅ Find part numbers and specifications
- ✅ Access all product manuals in one place

---

**Version:** 1.0  
**Last Updated:** October 22, 2025  
**Status:** Production Ready


