import { Keypair } from '@solana/web3.js';
import fs from 'fs';
import crypto from 'crypto';

// const walletData = JSON.parse(fs.readFileSync('config/wallets.json', 'utf8'));
// console.log(Buffer.from(walletData[0].privateKey, 'base64').toString('hex'));

// try {
//   const walletData = JSON.parse(fs.readFileSync('config/wallets.json'));
//   const keypair = Keypair.fromSecretKey(Uint8Array.from(walletData));
//   console.log('Wallet public key:', keypair.publicKey.toBase58());
// } catch (error) {
//   console.error('Wallet load error:', error.message);
// }


console.log(crypto.randomBytes(32).toString('hex')); // Save this to .env