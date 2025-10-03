#!/bin/bash

echo "🚀 Installing VBMM Bot Production..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Create directory structure
mkdir -p config logs data backups

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install tensorflow transformers pandas numpy aiohttp scikit-learn torch

# Make scripts executable
chmod +x vbmm-bot.js
chmod +x vbmm-ml.py

# Run setup
echo "🛠️  Running initial setup..."
node vbmm-bot.js setup --wallets 5

echo "✅ Installation completed!"
echo "📁 Next steps:"
echo "   1. Edit config/wallets.json with your actual private keys"
echo "   2. Fund your wallets with SOL"
echo "   3. Run: node vbmm-bot.js init -c config/trading-config.json"
echo "   4. Test with: node vbmm-bot.js trade -t TOKEN_MINT -a 100000"
