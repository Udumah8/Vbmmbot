# 1. Installation
chmod +x install.sh
./install.sh

# 2. Initialize with your funded wallets
node vbmm-bot.js init -c config/trading-config.json

# 3. Test single trade
node vbmm-bot.js trade -t EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v -a 1000000

# 4. Start volume boosting strategy
node vbmm-bot.js start-strategy -m volume -t TOKEN_MINT -a 500000

# 5. Execute holder growth
node vbmm-bot.js holders -t TOKEN_MINT -a 10000000 -c 50

# 6. Monitor metrics
node vbmm-bot.js metrics

# 7. Check wallet status
node vbmm-bot.js wallets
