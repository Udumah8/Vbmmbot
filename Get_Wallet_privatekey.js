import fs from 'fs';
import { Keypair } from '@solana/web3.js';
import bot from './v2.js'
import dotenv from 'dotenv';

// Load ENCRYPTION_KEY from .env

dotenv.config();
if (!process.env.ENCRYPTION_KEY) {
    console.error('❌ ENCRYPTION_KEY not found in .env');
    process.exit(1);
}

try {
    // Load wallets.json
    const walletData = JSON.parse(fs.readFileSync('D:/PythonProject/pythonProject/vbmm/config/wallets.json', 'utf8'));
    const botInstance = new bot();

    walletData.forEach((wallet, index) => {
        try {
            // Decrypt private key
            const decrypted = botInstance.decrypt(wallet.privateKey);
            const keypair = Keypair.fromSecretKey(Buffer.from(decrypted, 'base64'));
            const privateKeyBase64 = Buffer.from(keypair.secretKey).toString('base64');
            console.log(`Wallet ${index}:`);
            console.log(`  Public Key: ${keypair.publicKey.toBase58()}`);
            console.log(`  Private Key (base64): ${privateKeyBase64}`);
            console.log('  ⚠️ WARNING: Keep this private key secure! Do not share it.');
        } catch (error) {
            console.error(`❌ Failed to process Wallet ${index}: ${error.message}`);
        }
    });
} catch (error) {
    console.error('❌ Error reading wallets.json:', error.message);
}