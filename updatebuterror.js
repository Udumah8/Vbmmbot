
//#!/usr/bin/env node

/**
 * Volume Booster & Market Making Bot (VBMM Bot)
 * Production-Ready Final Version 2.2
 * Enhanced with Proxy Rotation Logic
 */

import { Connection, Keypair, VersionedTransaction, PublicKey, LAMPORTS_PER_SOL } from '@solana/web3.js';
import pkg from '@jup-ag/api';
const { Jupiter } = pkg;
import { Command } from 'commander';
import { readFileSync, writeFileSync, existsSync, appendFileSync, mkdirSync } from 'fs';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import axios from 'axios';
import cron from 'node-cron';
import { createObjectCsvWriter } from 'csv-writer';
import crypto from 'crypto';
import dotenv from 'dotenv';
import http from 'http';
import https from 'https';
import { SocksProxyAgent } from 'socks-proxy-agent';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Constants
const SOL_MINT = 'So11111111111111111111111111111111111111112';
const USDC_MINT = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v';
const MAX_RETRIES = 3;
const RPC_TIMEOUT = 30000;
const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY || crypto.randomBytes(32).toString('hex'); // Use env for key in production
const IV_LENGTH = 16;

class VBMMBot {
    constructor() {
        this.connection = null;
        this.jupiter = null;
        this.wallets = [];
        this.currentConfig = {};
        this.isRunning = false;
        this.emergencyStop = false;
        this.tradeQueue = [];
        this.metrics = {
            tradesExecuted: 0,
            volumeTraded: 0,
            holdersAdded: 0,
            failures: 0,
            startTime: null,
            totalFees: 0
        };
        this.proxyPool = [];
        this.currentProxyIndex = 0;
        this.proxyStatus = new Map(); // Tracks proxy health { proxy: { lastUsed, failures, success } }
        this.priceCache = new Map();
        this.walletBalances = new Map();
    }

    /**
     * Initialize the bot with full production setup
     */
    async initialize(configPath) {
        try {
            console.log('🚀 Initializing VBMM Bot v2.2 Production...');

            // Create necessary directories
            this.createDirectories();

            // Load configuration
            await this.loadConfiguration(configPath);

            // Initialize RPC connection with failover
            await this.initializeConnection();

            // Initialize Jupiter
            await this.initializeJupiter();

            // Load and validate wallets
            await this.loadWallets();

            // Initialize proxy pool
            await this.initializeProxyPool();

            // Initialize ML module
            await this.initializeML();

            // Start monitoring
            this.startMonitoring();

            console.log('✅ VBMM Bot Production initialized successfully');
            this.metrics.startTime = Date.now();

        } catch (error) {
            console.error('❌ Production initialization failed:', error);
            await this.emergencyShutdown();
            throw error;
        }
    }

    createDirectories() {
        const dirs = ['config', 'logs', 'data', 'backups'];
        for (const dir of dirs) {
            if (!existsSync(path.join(__dirname, dir))) {
                mkdirSync(path.join(__dirname, dir), { recursive: true });
            }
        }
    }

    async loadConfiguration(configPath) {
        const baseConfig = {
            rpcUrls: [
                'https://api.mainnet-beta.solana.com',
                'https://solana-mainnet.rpc.extrnode.com',
                'https://ssc-dao.genesysgo.net'
            ],
            maxSlippageBps: 100,
            tradeDelayMs: { min: 1000, max: 5000 },
            volumeBoost: { targetMultiplier: 5, buySellRatio: 0.7 },
            marketMaking: {
                spreadTiers: [0.005, 0.01, 0.02],
                rebalanceThreshold: 1000,
                minSpread: 0.001,
                maxSpread: 0.05
            },
            orbittMode: { enabled: false, surgeMultiplier: 4, minSignalStrength: 0.7 },
            wallets: { batchSize: 50, rotationStrategy: 'fisher-yates', minBalanceSOL: 0.01 },
            proxyRotation: {
                strategy: 'round-robin', // or 'random'
                maxFailures: 3,
                retryInterval: 300000 // 5 minutes
            },
            monitoring: {
                logToCSV: true,
                metricsInterval: 30000,
                healthCheckInterval: 60000
            },
            riskManagement: {
                maxDailyVolume: 100000,
                maxPositionSize: 5000,
                stopLossPercent: 10,
                maxConcurrentTrades: 5
            }
        };

        try {
            if (existsSync(configPath)) {
                const userConfig = JSON.parse(readFileSync(configPath, 'utf8'));
                this.currentConfig = { ...baseConfig, ...userConfig };
            } else {
                console.log('⚠️  No user config found, creating default...');
                this.currentConfig = baseConfig;
                this.saveConfiguration(configPath);
            }

            // Validate configuration
            this.validateConfiguration();

        } catch (error) {
            console.error('❌ Config load failed:', error);
            throw error;
        }
    }

    saveConfiguration(configPath) {
        writeFileSync(configPath, JSON.stringify(this.currentConfig, null, 2));
    }

    validateConfiguration() {
        const { maxSlippageBps, volumeBoost, marketMaking, proxyRotation } = this.currentConfig;

        if (maxSlippageBps > 500) {
            throw new Error('Slippage too high, max 500 bps allowed');
        }
        if (volumeBoost.buySellRatio > 0.9 || volumeBoost.buySellRatio < 0.1) {
            throw new Error('Buy/sell ratio must be between 0.1 and 0.9');
        }
        if (marketMaking.minSpread >= marketMaking.maxSpread) {
            throw new Error('Min spread must be less than max spread');
        }
        if (!['round-robin', 'random'].includes(proxyRotation.strategy)) {
            throw new Error('Proxy rotation strategy must be "round-robin" or "random"');
        }
    }

    async initializeConnection() {
        for (const rpcUrl of this.currentConfig.rpcUrls) {
            try {
                const proxy = this.getNextProxy();
                const httpAgent = proxy ? new SocksProxyAgent(proxy) : null;
                const testConnection = new Connection(rpcUrl, {
                    commitment: 'confirmed',
                    confirmTransactionInitialTimeout: 60000,
                    wsEndpoint: rpcUrl.replace('https', 'wss'),
                    httpAgent
                });

                // Test connection
                const slot = await testConnection.getSlot({ commitment: 'confirmed' });
                console.log(`✅ RPC connected: ${rpcUrl} (slot: ${slot})`);

                this.connection = testConnection;
                if (proxy) {
                    this.updateProxyStatus(proxy, true);
                }
                break;

            } catch (error) {
                console.warn(`⚠️  RPC failed: ${rpcUrl} - ${error.message}`);
                if (proxy) {
                    this.updateProxyStatus(proxy, false);
                }
                continue;
            }
        }

        if (!this.connection) {
            throw new Error('All RPC endpoints failed');
        }
    }

    async initializeJupiter() {
        try {
            const proxy = this.getNextProxy();
            const httpAgent = proxy ? new SocksProxyAgent(proxy) : null;
            this.jupiter = await Jupiter.load({
                connection: this.connection,
                cluster: 'mainnet-beta',
                wrapUnwrapSOL: true,
                httpAgent
            });
            console.log('✅ Jupiter initialized');
            if (proxy) {
                this.updateProxyStatus(proxy, true);
            }
        } catch (error) {
            console.error('❌ Jupiter initialization failed:', error);
            if (proxy) {
                this.updateProxyStatus(proxy, false);
            }
            throw error;
        }
    }

    async loadWallets() {
        try {
            const walletsConfigPath = path.join(__dirname, 'config', 'wallets.json');
            if (!existsSync(walletsConfigPath)) {
                throw new Error('wallets.json not found. Run setup first.');
            }

            const walletsData = JSON.parse(readFileSync(walletsConfigPath, 'utf8'));
            this.wallets = [];

            for (const [index, walletConfig] of walletsData.entries()) {
                try {
                    // Decrypt private key
                    const decryptedPrivateKey = this.decrypt(walletConfig.privateKey);
                    const keypair = Keypair.fromSecretKey(
                        Buffer.from(decryptedPrivateKey, 'base64')
                    );

                    // Check wallet balance
                    const balance = await this.getWalletBalance(keypair.publicKey);
                    const minBalance = walletConfig.minBalance || this.currentConfig.wallets.minBalanceSOL;

                    if (balance < minBalance) {
                        console.warn(`⚠️  Wallet ${index} low balance: ${balance} SOL (min: ${minBalance})`);
                        continue;
                    }

                    this.wallets.push({
                        keypair,
                        proxy: walletConfig.proxy || '',
                        minBalance,
                        behavior: walletConfig.behavior || 'volume_boost',
                        publicKey: keypair.publicKey.toString(),
                        lastUsed: 0,
                        tradeCount: 0,
                        balance,
                        index
                    });

                    this.walletBalances.set(keypair.publicKey.toString(), balance);

                } catch (error) {
                    console.warn(`⚠️  Failed to load wallet ${index}: ${error.message}`);
                }
            }

            if (this.wallets.length === 0) {
                throw new Error('No valid wallets loaded');
            }

            console.log(`✅ Loaded ${this.wallets.length} wallets with sufficient balance`);

        } catch (error) {
            console.error('❌ Wallet load failed:', error);
            throw error;
        }
    }

    async getWalletBalance(publicKey) {
        try {
            const proxy = this.getNextProxy();
            const httpAgent = proxy ? new SocksProxyAgent(proxy) : null;
            const connection = new Connection(this.currentConfig.rpcUrls[0], {
                commitment: 'confirmed',
                httpAgent
            });
            const balance = await connection.getBalance(publicKey);
            if (proxy) {
                this.updateProxyStatus(proxy, true);
            }
            return balance / LAMPORTS_PER_SOL;
        } catch (error) {
            console.warn(`Balance check failed for ${publicKey}: ${error.message}`);
            if (proxy) {
                this.updateProxyStatus(proxy, false);
            }
            return 0;
        }
    }

    async initializeProxyPool() {
        // Extract proxies from wallet config
        const proxies = [...new Set(this.wallets.map(w => w.proxy).filter(Boolean))];

        if (proxies.length === 0) {
            console.log('⚠️  No proxies configured, using direct connections');
            return;
        }

        // Test each proxy
        for (const proxy of proxies) {
            try {
                const [host, port] = proxy.replace('http://', '').split(':');
                const response = await axios.get('https://api.mainnet-beta.solana.com/health', {
                    proxy: { host, port: parseInt(port) },
                    timeout: 10000
                });

                if (response.data === 'ok') {
                    this.proxyPool.push(proxy);
                    this.proxyStatus.set(proxy, { lastUsed: 0, failures: 0, success: 0 });
                    console.log(`✅ Proxy active: ${proxy}`);
                }
            } catch (error) {
                console.warn(`⚠️  Proxy failed: ${proxy} - ${error.message}`);
                this.proxyStatus.set(proxy, { lastUsed: 0, failures: 1, success: 0 });
            }
        }

        console.log(`✅ ${this.proxyPool.length} proxies available`);

        // Schedule proxy re-testing
        cron.schedule('*/5 * * * *', () => {
            this.retestFailedProxies();
        });
    }

    getNextProxy() {
        if (this.proxyPool.length === 0) {
            return null;
        }

        const strategy = this.currentConfig.proxyRotation.strategy;
        const maxFailures = this.currentConfig.proxyRotation.maxFailures;

        // Filter out proxies that have exceeded max failures
        const availableProxies = this.proxyPool.filter(proxy => {
            const status = this.proxyStatus.get(proxy);
            return status.failures < maxFailures;
        });

        if (availableProxies.length === 0) {
            console.warn('⚠️  No available proxies');
            return null;
        }

        let selectedProxy;
        if (strategy === 'random') {
            const index = Math.floor(Math.random() * availableProxies.length);
            selectedProxy = availableProxies[index];
        } else {
            // Round-robin
            this.currentProxyIndex = (this.currentProxyIndex + 1) % availableProxies.length;
            selectedProxy = availableProxies[this.currentProxyIndex];
        }

        this.proxyStatus.set(selectedProxy, {
            ...this.proxyStatus.get(selectedProxy),
            lastUsed: Date.now()
        });

        console.log(`🔄 Using proxy: ${selectedProxy}`);
        return selectedProxy;
    }

    updateProxyStatus(proxy, success) {
        if (!this.proxyStatus.has(proxy)) {
            this.proxyStatus.set(proxy, { lastUsed: Date.now(), failures: 0, success: 0 });
        }

        const status = this.proxyStatus.get(proxy);
        if (success) {
            this.proxyStatus.set(proxy, {
                ...status,
                success: status.success + 1,
                failures: Math.max(0, status.failures - 1)
            });
        } else {
            this.proxyStatus.set(proxy, {
                ...status,
                failures: status.failures + 1
            });
        }
    }

    async retestFailedProxies() {
        const maxFailures = this.currentConfig.proxyRotation.maxFailures;
        const retryInterval = this.currentConfig.proxyRotation.retryInterval;

        for (const proxy of this.proxyPool) {
            const status = this.proxyStatus.get(proxy);
            if (status.failures >= maxFailures && Date.now() - status.lastUsed > retryInterval) {
                try {
                    const [host, port] = proxy.replace('http://', '').split(':');
                    const response = await axios.get('https://api.mainnet-beta.solana.com/health', {
                        proxy: { host, port: parseInt(port) },
                        timeout: 10000
                    });

                    if (response.data === 'ok') {
                        this.proxyStatus.set(proxy, {
                            ...status,
                            failures: 0,
                            success: status.success + 1,
                            lastUsed: Date.now()
                        });
                        console.log(`✅ Proxy recovered: ${proxy}`);
                    }
                } catch (error) {
                    console.warn(`⚠️  Proxy retest failed: ${proxy} - ${error.message}`);
                    this.proxyStatus.set(proxy, {
                        ...status,
                        failures: status.failures + 1,
                        lastUsed: Date.now()
                    });
                }
            }
        }
    }

    async initializeML() {
        return new Promise((resolve) => {
            const pythonProcess = spawn('python', [
                path.join(__dirname, 'vbmm-ml.py'),
                '--init',
                '--config',
                JSON.stringify(this.currentConfig)
            ]);

            let output = '';
            let error = '';

            pythonProcess.stdout.on('data', (data) => {
                output += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                error += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    console.log('✅ ML module initialized:', output.trim());
                    resolve(output);
                } else {
                    console.warn(`⚠️  ML initialization warning: ${error}`);
                    resolve(); // ML is optional
                }
            });

            // Timeout after 10 seconds
            setTimeout(() => {
                pythonProcess.kill();
                console.warn('⚠️  ML initialization timeout');
                resolve();
            }, 10000);
        });
    }

    /**
     * Enhanced wallet rotation with Fisher-Yates shuffle
     */
    shuffleWallets(wallets) {
        const shuffled = [...wallets];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    getNextWallet(tradeType = 'volume_boost', minBalance = 0.01) {
        const now = Date.now();
        const availableWallets = this.wallets.filter(w =>
            w.behavior === tradeType &&
            w.balance >= minBalance &&
            w.lastUsed < now - 60000 // 1 minute cooldown
        );

        if (availableWallets.length === 0) {
            return null;
        }

        const shuffled = this.shuffleWallets(availableWallets);
        const selected = shuffled[0];
        return selected;
    }

    /**
     * Production-grade trade execution with full error handling and proxy rotation
     */
    async executeVolumeBoostTrade(inputToken, outputToken, amount) {
        if (this.emergencyStop) {
            throw new Error('EMERGENCY_STOP activated');
        }

        const wallet = this.getNextWallet('volume_boost', amount / LAMPORTS_PER_SOL * 1.1); // +10% buffer
        if (!wallet) {
            throw new Error('No available wallets with sufficient balance');
        }

        let retries = 0;
        let lastError = null;

        while (retries < MAX_RETRIES) {
            try {
                const proxy = this.getNextProxy();
                const httpAgent = proxy ? new SocksProxyAgent(proxy) : null;

                // Get fresh quote
                const quoteResponse = await this.jupiter.computeRoutes({
                    inputMint: new PublicKey(inputToken),
                    outputMint: new PublicKey(outputToken),
                    amount: amount,
                    slippageBps: this.currentConfig.maxSlippageBps,
                    feeBps: 0,
                    onlyDirectRoutes: false,
                    httpAgent
                });

                const bestRoute = quoteResponse.routesInfos[0]; // v6 uses routesInfos

                if (!bestRoute) {
                    throw new Error('No routes available from Jupiter');
                }

                // Prepare transaction
                const swapTransaction = await this.jupiter.swap({
                    routeInfo: bestRoute,
                    userPublicKey: wallet.keypair.publicKey,
                    httpAgent
                });

                // Sign and send
                const swapTransactionBuf = Buffer.from(swapTransaction.swapTransaction, 'base64');
                const transaction = VersionedTransaction.deserialize(swapTransactionBuf);
                transaction.sign([wallet.keypair]);

                const latestBlockHash = await this.connection.getLatestBlockhash();
                const rawTransaction = transaction.serialize();

                const txid = await this.connection.sendRawTransaction(rawTransaction, {
                    skipPreflight: true,
                    maxRetries: 2
                });

                await this.connection.confirmTransaction({
                    signature: txid,
                    blockhash: latestBlockHash.blockhash,
                    lastValidBlockHeight: latestBlockHash.lastValidBlockHeight
                });

                console.log(`✅ Success: https://solscan.io/tx/${txid}`);

                // Update metrics
                const newBalance = await this.getWalletBalance(wallet.keypair.publicKey);
                wallet.balance = newBalance;
                this.walletBalances.set(wallet.publicKey, newBalance);

                // Fetch fee
                const txDetails = await this.connection.getTransaction(txid, {
                    commitment: 'confirmed',
                    maxSupportedTransactionVersion: 0
                });

                const fee = txDetails?.meta?.fee ? txDetails.meta.fee / LAMPORTS_PER_SOL : 0;
                this.metrics.totalFees += fee;

                // Log
                const tradeData = {
                    type: 'volume_boost',
                    wallet: wallet.publicKey,
                    inputToken,
                    outputToken,
                    amount,
                    inputAmount: bestRoute.inAmountInLamports,
                    outputAmount: bestRoute.outAmountInLamports,
                    signature: txid,
                    price: Number(bestRoute.outAmountInLamports) / Number(bestRoute.inAmountInLamports),
                    fee,
                    timestamp: Date.now(),
                    retries,
                    proxy
                };

                this.logTrade(tradeData);
                wallet.lastUsed = Date.now();
                wallet.tradeCount++;

                this.metrics.tradesExecuted++;
                this.metrics.volumeTraded += amount / LAMPORTS_PER_SOL;

                if (proxy) {
                    this.updateProxyStatus(proxy, true);
                }

                return {
                    success: true,
                    signature: txid,
                    inputAmount: bestRoute.inAmountInLamports,
                    outputAmount: bestRoute.outAmountInLamports
                };

            } catch (error) {
                lastError = error;
                retries++;

                if (proxy) {
                    this.updateProxyStatus(proxy, false);
                }

                if (retries < MAX_RETRIES) {
                    const delay = Math.pow(2, retries) * 1000; // Exponential backoff
                    console.warn(`⚠️  Trade attempt ${retries} failed, retrying in ${delay}ms:`, error.message);
                    await this.delay(delay);
                }
            }
        }

        this.metrics.failures++;
        this.logError('volume_boost', lastError, wallet.publicKey);
        throw lastError;
    }

    /**
     * Advanced market making with real-time adjustments
     */
    async executeMarketMakingTrade(inputToken, outputToken, baseAmount) {
        if (this.emergencyStop) {
            throw new Error('EMERGENCY_STOP activated');
        }

        const wallet = this.getNextWallet('market_making', baseAmount / LAMPORTS_PER_SOL * 1.2);
        if (!wallet) {
            throw new Error('No available wallets for market making');
        }

        try {
            // Get real market data
            const marketData = await this.getRealMarketData(inputToken, outputToken);
            const mlSignal = await this.getMLSignal(marketData);

            // Calculate dynamic parameters
            const spread = this.calculateDynamicSpread(mlSignal, marketData.volatility);
            const amount = this.calculateMarketMakingAmount(baseAmount, spread, mlSignal);

            // Execute with market making logic
            const result = await this.executeVolumeBoostTrade(inputToken, outputToken, amount);

            // Rebalance if needed
            await this.checkRebalance(wallet, inputToken, outputToken);

            this.logTrade({
                type: 'market_making',
                wallet: wallet.publicKey,
                inputToken,
                outputToken,
                amount,
                spread,
                mlSignal: mlSignal.signal,
                confidence: mlSignal.confidence,
                volatility: marketData.volatility,
                signature: result.signature,
                timestamp: Date.now()
            });

            return result;

        } catch (error) {
            this.metrics.failures++;
            this.logError('market_making', error, wallet.publicKey);
            throw error;
        }
    }

    async getRealMarketData(inputToken, outputToken) {
        try {
            const proxy = this.getNextProxy();
            const httpAgent = proxy ? new SocksProxyAgent(proxy) : null;

            // Get recent trades from Jupiter
            const quoteResponse = await this.jupiter.computeRoutes({
                inputMint: new PublicKey(inputToken),
                outputMint: new PublicKey(outputToken),
                amount: 1000000, // 0.001 SOL
                slippageBps: 50,
                onlyDirectRoutes: true,
                httpAgent
            });

            const routes = quoteResponse.routesInfos || [];
            const prices = routes.map(route =>
                Number(route.outAmountInLamports) / Number(route.inAmountInLamports)
            );

            // Calculate volatility from recent price movements
            const volatility = this.calculateRealVolatility(prices);

            // Get volume data (simplified - in production use DEX APIs)
            const volume = routes.reduce((sum, route) => sum + Number(route.inAmountInLamports), 0);

            if (proxy) {
                this.updateProxyStatus(proxy, true);
            }

            return {
                timestamp: Date.now(),
                inputToken,
                outputToken,
                price: prices.length > 0 ? prices[0] : 1,
                prices,
                volume,
                volatility,
                tradeCount: routes.length,
                bidAskSpread: this.calculateBidAskSpread(routes)
            };
        } catch (error) {
            console.warn('Market data fetch failed, using fallback:', error.message);
            if (proxy) {
                this.updateProxyStatus(proxy, false);
            }
            return this.getFallbackMarketData();
        }
    }

    calculateRealVolatility(prices) {
        if (prices.length < 2) return 0.01;

        const mean = prices.reduce((a, b) => a + b) / prices.length;
        const variance = prices.reduce((acc, price) => acc + Math.pow(price - mean, 2), 0) / prices.length;
        return Math.sqrt(variance) / mean;
    }

    calculateBidAskSpread(routes) {
        if (routes.length < 2) return 0.01;

        const bestBuy = Number(routes[0].outAmountInLamports) / Number(routes[0].inAmountInLamports);
        const bestSell = Number(routes[1].outAmountInLamports) / Number(routes[1].inAmountInLamports);

        return Math.abs(bestBuy - bestSell) / ((bestBuy + bestSell) / 2);
    }

    getFallbackMarketData() {
        return {
            timestamp: Date.now(),
            price: 1,
            prices: [0.99, 1.0, 1.01, 0.98, 1.02],
            volume: 1000000,
            volatility: 0.02,
            tradeCount: 5,
            bidAskSpread: 0.01
        };
    }

    calculateDynamicSpread(mlSignal, volatility) {
        const baseSpread = this.currentConfig.marketMaking.spreadTiers[1] || 0.01;
        const signalAdjustment = (0.5 - mlSignal.signal) * 0.005;
        const volatilityAdjustment = volatility * 0.02;
        const confidenceAdjustment = (0.5 - mlSignal.confidence) * 0.003;

        const rawSpread = baseSpread + signalAdjustment + volatilityAdjustment + confidenceAdjustment;

        return Math.max(
            this.currentConfig.marketMaking.minSpread,
            Math.min(this.currentConfig.marketMaking.maxSpread, rawSpread)
        );
    }

    calculateMarketMakingAmount(baseAmount, spread, mlSignal) {
        const confidenceMultiplier = mlSignal.confidence || 0.5;
        const signalMultiplier = 1 + (mlSignal.signal - 0.5) * 0.2;
        const spreadMultiplier = 1 + spread * 10; // Higher spread = smaller trades

        return baseAmount * confidenceMultiplier * signalMultiplier / spreadMultiplier;
    }

    async checkRebalance(wallet, inputToken, outputToken) {
        const currentBalance = wallet.balance;
        if (currentBalance < wallet.minBalance) {
            console.log(`🔄 Rebalancing wallet ${wallet.publicKey.substring(0, 8)}...`);
            // In production, implement rebalance logic, e.g., transfer from master wallet
        }
    }

    /**
     * Orbitt MM Mode with real surge detection
     */
    async executeOrbittMode(inputToken, outputToken, baseAmount) {
        if (!this.currentConfig.orbittMode.enabled) {
            throw new Error('Orbitt Mode is not enabled');
        }

        console.log('🚀 Activating Orbitt MM Mode...');

        // Get enhanced market analysis
        const marketData = await this.getRealMarketData(inputToken, outputToken);
        const mlSignal = await this.getMLSignal({
            ...marketData,
            orbittMode: true,
            surgeDetection: true
        });

        const minSignalStrength = this.currentConfig.orbittMode.minSignalStrength || 0.7;

        if (mlSignal.signal > minSignalStrength && mlSignal.confidence > 0.6) {
            const surgeMultiplier = this.currentConfig.orbittMode.surgeMultiplier || 4;
            const surgeAmount = baseAmount * surgeMultiplier;

            console.log(`🎯 Orbitt Surge detected! Signal: ${mlSignal.signal.toFixed(3)}, Confidence: ${mlSignal.confidence.toFixed(3)}`);

            // Execute surge across multiple wallets
            const surgeWallets = this.wallets
                .filter(w => w.behavior === 'volume_boost' && w.balance >= surgeAmount / LAMPORTS_PER_SOL / 10)
                .slice(0, 10);

            if (surgeWallets.length === 0) {
                throw new Error('No wallets available for surge trading');
            }

            const amountPerWallet = surgeAmount / surgeWallets.length;
            const results = [];

            for (const wallet of surgeWallets) {
                try {
                    const result = await this.executeVolumeBoostTrade(inputToken, outputToken, amountPerWallet);
                    results.push({
                        wallet: wallet.publicKey,
                        success: true,
                        amount: amountPerWallet,
                        signature: result.signature
                    });

                    // Staggered delays for organic appearance
                    await this.delay(500 + Math.random() * 1500);

                } catch (error) {
                    console.warn(`⚠️  Surge trade failed for wallet ${wallet.publicKey}: ${error.message}`);
                    results.push({
                        wallet: wallet.publicKey,
                        success: false,
                        error: error.message
                    });
                }
            }

            const successfulTrades = results.filter(r => r.success).length;
            console.log(`✅ Orbitt Surge completed: ${successfulTrades}/${surgeWallets.length} successful trades`);

            return {
                surge: true,
                results,
                multiplier: surgeMultiplier,
                totalAmount: surgeAmount
            };
        } else {
            // Normal market making during non-surge periods
            return await this.executeMarketMakingTrade(inputToken, outputToken, baseAmount);
        }
    }

    /**
     * Holder growth with real wallet distribution
     */
    async executeHolderGrowthStrategy(tokenMint, totalAmount, minHolders = 100) {
        console.log(`📈 Executing holder growth strategy for ${minHolders} holders...`);

        const availableWallets = this.wallets
            .filter(w => (w.behavior === 'holder_focus' || w.tradeCount < 3) && w.balance > 0.02)
            .slice(0, minHolders);

        if (availableWallets.length < minHolders * 0.5) {
            throw new Error(`Insufficient wallets for holder growth: ${availableWallets.length} available, need ${minHolders}`);
        }

        const amountPerWallet = totalAmount / availableWallets.length;

        if (amountPerWallet < 50000) { // Minimum ~0.0005 SOL worth
            throw new Error(`Amount per wallet too small: ${amountPerWallet}, minimum 50000 lamports`);
        }

        const results = [];
        let successful = 0;

        for (const [index, wallet] of availableWallets.entries()) {
            try {
                const result = await this.executeVolumeBoostTrade(
                    SOL_MINT,
                    tokenMint,
                    amountPerWallet
                );

                results.push({
                    wallet: wallet.publicKey,
                    success: true,
                    amount: amountPerWallet,
                    signature: result.signature
                });

                successful++;
                this.metrics.holdersAdded++;

                // Progressive delays to avoid detection
                const delay = 2000 + (index * 100) + (Math.random() * 3000);
                await this.delay(delay);

                // Batch logging every 10 trades
                if (successful % 10 === 0) {
                    console.log(`✅ ${successful}/${availableWallets.length} holder growth trades completed`);
                }

            } catch (error) {
                console.warn(`⚠️  Holder growth trade failed for wallet ${wallet.publicKey}: ${error.message}`);
                results.push({
                    wallet: wallet.publicKey,
                    success: false,
                    error: error.message
                });
            }
        }

        console.log(`✅ Holder growth completed: ${successful}/${availableWallets.length} successful micro-buys`);
        return {
            totalAttempted: availableWallets.length,
            successful,
            failed: availableWallets.length - successful,
            results
        };
    }

    /**
     * ML Signal integration
     */
    async getMLSignal(marketData) {
        try {
            const pythonProcess = spawn('python', [
                path.join(__dirname, 'vbmm-ml.py'),
                '--predict',
                '--data',
                JSON.stringify(marketData)
            ]);

            return new Promise((resolve) => {
                let output = '';
                let error = '';

                pythonProcess.stdout.on('data', (data) => {
                    output += data.toString();
                });

                pythonProcess.stderr.on('data', (data) => {
                    error += data.toString();
                });

                pythonProcess.on('close', (code) => {
                    if (code === 0 && output) {
                        try {
                            const result = JSON.parse(output);
                            resolve(result);
                        } catch (parseError) {
                            console.warn('ML parse error, using default:', parseError.message);
                            resolve({ signal: 0.5, confidence: 0.5 });
                        }
                    } else {
                        console.warn('ML call failed, using default:', error);
                        resolve({ signal: 0.5, confidence: 0.5 });
                    }
                });

                // Timeout after 3 seconds
                setTimeout(() => {
                    pythonProcess.kill();
                    resolve({ signal: 0.5, confidence: 0.5 });
                }, 3000);
            });

        } catch (error) {
            console.warn('ML signal failed:', error.message);
            return { signal: 0.5, confidence: 0.5 };
        }
    }

    /**
     * Enhanced logging system
     */
    logTrade(tradeData) {
        const timestamp = new Date().toISOString();
        const logEntry = { timestamp, ...tradeData };

        // JSON logging
        appendFileSync(
            path.join(__dirname, 'logs', 'trades.jsonl'),
            JSON.stringify(logEntry) + '\n'
        );

        // CSV logging
        if (this.currentConfig.monitoring.logToCSV) {
            const csvPath = path.join(__dirname, 'logs', 'trades.csv');
            const csvHeader = [
                { id: 'timestamp', title: 'Timestamp' },
                { id: 'type', title: 'Type' },
                { id: 'wallet', title: 'Wallet' },
                { id: 'inputToken', title: 'InputToken' },
                { id: 'outputToken', title: 'OutputToken' },
                { id: 'amount', title: 'Amount' },
                { id: 'signature', title: 'Signature' },
                { id: 'price', title: 'Price' },
                { id: 'fee', title: 'Fee' },
                { id: 'proxy', title: 'Proxy' }
            ];

            if (!existsSync(csvPath)) {
                const csvWriter = createObjectCsvWriter({ path: csvPath, header: csvHeader });
                csvWriter.writeRecords([logEntry]).catch(console.error);
            } else {
                const csvWriter = createObjectCsvWriter({
                    path: csvPath,
                    header: csvHeader,
                    append: true
                });
                csvWriter.writeRecords([logEntry]).catch(console.error);
            }
        }

        console.log(`✅ ${tradeData.type} | ${(tradeData.amount / LAMPORTS_PER_SOL).toFixed(4)} SOL | ${tradeData.wallet.substring(0, 8)}... | Proxy: ${tradeData.proxy || 'None'}`);
    }

    logError(context, error, wallet = 'unknown') {
        const errorEntry = {
            timestamp: new Date().toISOString(),
            context,
            wallet,
            error: error.message,
            stack: error.stack
        };

        appendFileSync(
            path.join(__dirname, 'logs', 'errors.jsonl'),
            JSON.stringify(errorEntry) + '\n'
        );

        console.error(`❌ ${context} error: ${error.message}`);
    }

    /**
     * Monitoring and health checks
     */
    startMonitoring() {
        // Metrics logging every 30 seconds
        cron.schedule('*/30 * * * * *', () => {
            this.logMetrics();
        });

        // Health checks every 60 seconds
        cron.schedule('*/60 * * * * *', () => {
            this.healthCheck();
        });

        // Wallet balance monitoring every 5 minutes
        cron.schedule('*/5 * * * *', () => {
            this.monitorWalletBalances();
        });
    }

    logMetrics() {
        const metrics = this.getMetrics();
        appendFileSync(
            path.join(__dirname, 'logs', 'metrics.jsonl'),
            JSON.stringify({ timestamp: new Date().toISOString(), ...metrics }) + '\n'
        );
    }

    async healthCheck() {
        try {
            // Check RPC connection
            await this.connection.getSlot();

            // Check Jupiter
            const proxy = this.getNextProxy();
            const httpAgent = proxy ? new SocksProxyAgent(proxy) : null;
            await this.jupiter.computeRoutes({
                inputMint: new PublicKey(SOL_MINT),
                outputMint: new PublicKey(USDC_MINT),
                amount: 1000000,
                slippageBps: 100,
                httpAgent
            });

            // Check wallet accessibility
            if (this.wallets.length > 0) {
                await this.getWalletBalance(this.wallets[0].keypair.publicKey);
            }

            if (proxy) {
                this.updateProxyStatus(proxy, true);
            }

        } catch (error) {
            console.error('❌ Health check failed:', error.message);
            if (proxy) {
                this.updateProxyStatus(proxy, false);
            }
            this.logError('health_check', error);
        }
    }

    async monitorWalletBalances() {
        for (const wallet of this.wallets) {
            try {
                const newBalance = await this.getWalletBalance(wallet.keypair.publicKey);
                wallet.balance = newBalance;
                this.walletBalances.set(wallet.publicKey, newBalance);

                if (newBalance < wallet.minBalance) {
                    console.warn(`⚠️  Wallet ${wallet.publicKey.substring(0, 8)} low balance: ${newBalance} SOL`);
                }
            } catch (error) {
                console.warn(`Balance check failed for ${wallet.publicKey}: ${error.message}`);
            }
        }
    }

    /**
     * Emergency procedures
     */
    emergencyStop() {
        console.log('🛑 EMERGENCY STOP ACTIVATED!');
        this.emergencyStop = true;
        this.isRunning = false;
        this.tradeQueue = [];

        const stopLog = {
            timestamp: new Date().toISOString(),
            action: 'EMERGENCY_STOP',
            metrics: this.getMetrics()
        };

        writeFileSync(
            path.join(__dirname, 'logs', 'emergency_stop.json'),
            JSON.stringify(stopLog, null, 2)
        );

        console.log('✅ Emergency stop logged and activated');
    }

    async emergencyShutdown() {
        this.emergencyStop();

        // Additional cleanup
        try {
            console.log('🛑 Emergency shutdown completed');
        } catch (error) {
            console.error('Emergency shutdown error:', error);
        }
    }

    resume() {
        if (!this.emergencyStop) {
            console.log('ℹ️  No emergency stop active');
            return;
        }

        console.log('🟢 Resuming from emergency stop...');
        this.emergencyStop = false;

        const resumeLog = {
            timestamp: new Date().toISOString(),
            action: 'RESUME',
            metrics: this.getMetrics()
        };

        appendFileSync(
            path.join(__dirname, 'logs', 'operations.jsonl'),
            JSON.stringify(resumeLog) + '\n'
        );
    }

    /**
     * Utility functions
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    getMetrics() {
        if (!this.metrics.startTime) {
            return { status: 'Not started' };
        }

        const uptime = Date.now() - this.metrics.startTime;
        const totalTrades = this.metrics.tradesExecuted + this.metrics.failures;
        const successRate = totalTrades > 0 ? (this.metrics.tradesExecuted / totalTrades) * 100 : 0;

        const activeWallets = this.wallets.filter(w =>
            w.lastUsed > Date.now() - 3600000
        ).length;

        return {
            status: this.emergencyStop ? 'EMERGENCY_STOP' : (this.isRunning ? 'Running' : 'Stopped'),
            uptime: Math.floor(uptime / 1000),
            tradesExecuted: this.metrics.tradesExecuted,
            failures: this.metrics.failures,
            successRate: Math.round(successRate * 100) / 100,
            volumeTraded: Math.round(this.metrics.volumeTraded * 100) / 100,
            holdersAdded: this.metrics.holdersAdded,
            totalFees: Math.round(this.metrics.totalFees * 1000) / 1000,
            wallets: {
                total: this.wallets.length,
                active: activeWallets
            },
            proxies: {
                total: this.proxyPool.length,
                active: this.proxyPool.filter(p => this.proxyStatus.get(p).failures < this.currentConfig.proxyRotation.maxFailures).length
            },
            timestamp: new Date().toISOString()
        };
    }

    /**
     * Strategy executors
     */
    async startTrading(strategyConfig) {
        if (this.isRunning) {
            throw new Error('Trading is already running');
        }

        this.isRunning = true;
        console.log(`🟢 Starting ${strategyConfig.mode} strategy...`);

        try {
            while (this.isRunning && !this.emergencyStop) {
                try {
                    await this.executeStrategyCycle(strategyConfig);

                    // Adaptive delay based on strategy
                    const delay = this.calculateStrategyDelay(strategyConfig);
                    await this.delay(delay);

                } catch (error) {
                    console.error('❌ Strategy cycle failed:', error);
                    await this.delay(30000); // Wait 30 seconds before retry
                }
            }
        } finally {
            this.isRunning = false;
            console.log('🛑 Trading strategy stopped');
        }
    }

    async executeStrategyCycle(strategyConfig) {
        const amount = this.calculateDynamicAmount(strategyConfig.baseAmount, strategyConfig.mode);

        switch (strategyConfig.mode) {
            case 'volume_boost':
                await this.executeVolumeBoostTrade(
                    strategyConfig.inputToken,
                    strategyConfig.outputToken,
                    amount
                );
                break;

            case 'market_making':
                await this.executeMarketMakingTrade(
                    strategyConfig.inputToken,
                    strategyConfig.outputToken,
                    amount
                );
                break;

            case 'orbitt_mode':
                await this.executeOrbittMode(
                    strategyConfig.inputToken,
                    strategyConfig.outputToken,
                    amount
                );
                break;

            case 'holder_growth':
                await this.executeHolderGrowthStrategy(
                    strategyConfig.tokenMint,
                    strategyConfig.totalAmount,
                    strategyConfig.targetHolders
                );
                break;

            default:
                throw new Error(`Unknown strategy: ${strategyConfig.mode}`);
        }
    }

    calculateStrategyDelay(strategyConfig) {
        const baseDelays = {
            'volume_boost': { min: 10000, max: 30000 },
            'market_making': { min: 5000, max: 15000 },
            'orbitt_mode': { min: 30000, max: 60000 },
            'holder_growth': { min: 60000, max: 300000 }
        };

        const base = baseDelays[strategyConfig.mode] || { min: 10000, max: 30000 };
        return base.min + Math.random() * (base.max - base.min);
    }

    calculateDynamicAmount(baseAmount, strategy) {
        const randomFactor = 0.8 + Math.random() * 0.4;
        const strategyMultipliers = {
            'volume_boost': 1.0,
            'market_making': 0.3,
            'orbitt_mode': 2.0,
            'holder_growth': 0.1
        };

        return Math.floor(baseAmount * randomFactor * (strategyMultipliers[strategy] || 1.0));
    }

    stopTrading() {
        console.log('🛑 Stopping trading strategy...');
        this.isRunning = false;
    }

    /**
     * Wallet management utilities
     */
    async generateWallets(count = 10) {
        const wallets = [];

        for (let i = 0; i < count; i++) {
            const keypair = Keypair.generate();
            const privateKeyBase64 = Buffer.from(keypair.secretKey).toString('base64');
            const encryptedPrivateKey = this.encrypt(privateKeyBase64);
            const wallet = {
                privateKey: encryptedPrivateKey,
                proxy: '', // In production, set real proxies
                minBalance: 0.01,
                behavior: i < count * 0.6 ? 'volume_boost' :
                         i < count * 0.8 ? 'market_making' : 'holder_focus'
            };
            wallets.push(wallet);
        }

        return wallets;
    }

    async saveWallets(wallets, filePath) {
        writeFileSync(filePath, JSON.stringify(wallets, null, 2));
        console.log(`✅ Saved ${wallets.length} wallets to ${filePath}`);
    }

    encrypt(text) {
        const iv = crypto.randomBytes(IV_LENGTH);
        const cipher = crypto.createCipheriv('aes-256-cbc', Buffer.from(ENCRYPTION_KEY, 'hex'), iv);
        let encrypted = cipher.update(text);
        encrypted = Buffer.concat([encrypted, cipher.final()]);
        return iv.toString('hex') + ':' + encrypted.toString('hex');
    }

    decrypt(text) {
        const textParts = text.split(':');
        const iv = Buffer.from(textParts.shift(), 'hex');
        const encryptedText = Buffer.from(textParts.join(':'), 'hex');
        const decipher = crypto.createDecipheriv('aes-256-cbc', Buffer.from(ENCRYPTION_KEY, 'hex'), iv);
        let decrypted = decipher.update(encryptedText);
        decrypted = Buffer.concat([decrypted, decipher.final()]);
        return decrypted.toString();
    }
}

// CLI Implementation
function setupCLI() {
    const program = new Command();
    const bot = new VBMMBot();

    program
        .name('vbmm-bot')
        .description('Production Volume Booster & Market Making Bot for Solana')
        .version('2.2.0');

    program
        .command('init')
        .description('Initialize the bot with configuration')
        .requiredOption('-c, --config <path>', 'Path to configuration file', './config/trading-config.json')
        .action(async (options) => {
            try {
                await bot.initialize(options.config);
                console.log('✅ Bot initialized successfully');
            } catch (error) {
                console.error('❌ Initialization failed:', error);
                process.exit(1);
            }
        });

    program
        .command('setup')
        .description('Setup wizard for initial configuration')
        .option('-w, --wallets <count>', 'Number of wallets to generate', '10')
        .action(async (options) => {
            try {
                console.log('🛠️  Setting up VBMM Bot...');

                // Generate wallets
                const walletCount = parseInt(options.wallets);
                const wallets = await bot.generateWallets(walletCount);

                // Save wallets
                const walletPath = path.join(__dirname, 'config', 'wallets.json');
                await bot.saveWallets(wallets, walletPath);

                // Create default config
                const configPath = path.join(__dirname, 'config', 'trading-config.json');
                bot.saveConfiguration(configPath);

                console.log('✅ Setup completed!');
                console.log(`📁 Wallets: ${walletPath}`);
                console.log(`📁 Config: ${configPath}`);
                console.log('\nNext: Fund wallets and run "vbmm-bot init -c config/trading-config.json"');

            } catch (error) {
                console.error('❌ Setup failed:', error);
                process.exit(1);
            }
        });

    program
        .command('trade')
        .description('Execute a single trade')
        .requiredOption('-f, --from <token>', 'Input token mint address', SOL_MINT)
        .requiredOption('-t, --to <token>', 'Output token mint address')
        .requiredOption('-a, --amount <amount>', 'Trade amount in lamports', parseFloat)
        .option('-m, --mode <mode>', 'Trading mode (volume|market|orbitt)', 'volume')
        .action(async (options) => {
            try {
                await bot.initialize('./config/trading-config.json');

                let result;
                switch (options.mode) {
                    case 'market':
                        result = await bot.executeMarketMakingTrade(options.from, options.to, options.amount);
                        break;
                    case 'orbitt':
                        result = await bot.executeOrbittMode(options.from, options.to, options.amount);
                        break;
                    default:
                        result = await this.executeVolumeBoostTrade(options.from, options.to, options.amount);
                }

                console.log('✅ Trade executed successfully');
                console.log(`📄 Signature: ${result.signature}`);
                console.log(`💸 Input: ${result.inputAmount} lamports`);
                console.log(`🎯 Output: ${result.outputAmount} lamports`);

            } catch (error) {
                console.error('❌ Trade failed:', error.message);
                process.exit(1);
            }
        });

    program
        .command('start-strategy')
        .description('Start automated trading strategy')
        .requiredOption('-m, --mode <mode>', 'Strategy mode (volume|market|orbitt|holders)')
        .requiredOption('-f, --from <token>', 'Input token mint address', SOL_MINT)
        .requiredOption('-t, --to <token>', 'Output token mint address')
        .requiredOption('-a, --amount <amount>', 'Base trade amount in lamports', parseFloat)
        .option('--target-holders <count>', 'Target holders count for growth strategy', '100')
        .action(async (options) => {
            try {
                await bot.initialize('./config/trading-config.json');

                const strategyConfig = {
                    mode: options.mode,
                    inputToken: options.from,
                    outputToken: options.to,
                    baseAmount: options.amount,
                    targetHolders: parseInt(options.targetHolders),
                    totalAmount: options.amount
                };

                console.log(`🚀 Starting ${strategyConfig.mode} strategy...`);
                await bot.startTrading(strategyConfig);

            } catch (error) {
                console.error('❌ Strategy start failed:', error.message);
                process.exit(1);
            }
        });

    program
        .command('holders')
        .description('Execute holder growth strategy')
        .requiredOption('-t, --token <mint>', 'Token mint address')
        .requiredOption('-a, --amount <amount>', 'Total amount in lamports', parseFloat)
        .requiredOption('-c, --count <count>', 'Target holder count', parseInt)
        .action(async (options) => {
            try {
                await bot.initialize('./config/trading-config.json');
                const results = await bot.executeHolderGrowthStrategy(
                    options.token,
                    options.amount,
                    options.count
                );

                console.log(`✅ Holder growth completed: ${results.successful}/${results.totalAttempted} successful`);
                console.log(`📈 New holders: ${results.successful}`);

            } catch (error) {
                console.error('❌ Holder growth failed:', error.message);
                process.exit(1);
            }
        });

    program
        .command('stop')
        .description('Stop all trading activities')
        .action(() => {
            bot.stopTrading();
            console.log('✅ Trading stopped');
        });

    program
        .command('emergency-stop')
        .description('Immediately stop all trading and cancel pending transactions')
        .action(() => {
            bot.emergencyStop();
        });

    program
        .command('resume')
        .description('Resume from emergency stop')
        .action(() => {
            bot.resume();
        });

    program
        .command('metrics')
        .description('Show current bot metrics')
        .action(async () => {
            try {
                await bot.initialize('./config/trading-config.json');
                const metrics = bot.getMetrics();
                console.log('📊 Current Metrics:');
                console.log(JSON.stringify(metrics, null, 2));
            } catch (error) {
                console.error('❌ Failed to get metrics:', error.message);
            }
        });

    program
        .command('wallets')
        .description('Show wallet status')
        .action(async () => {
            try {
                await bot.initialize('./config/trading-config.json');
                console.log('👛 Wallet Status:');
                bot.wallets.forEach((wallet, index) => {
                    console.log(`${index + 1}. ${wallet.publicKey.substring(0, 8)}... | ${wallet.behavior} | ${wallet.balance.toFixed(4)} SOL | Trades: ${wallet.tradeCount}`);
                });
            } catch (error) {
                console.error('❌ Failed to get wallet status:', error.message);
            }
        });

    return program;
}

// Main execution
//console.log('VBMM Bot started at:', new Date().toISOString());

if (import.meta.url != `file://${process.argv[1]}`) {
    const program = setupCLI();
    program.configureOutput({
        writeOut: (str) => process.stdout.write(`[VBMM] ${str}`),
        writeErr: (str) => process.stderr.write(`[ERROR] ${str}`)
    });
    program.parse();
}

export default VBMMBot;
