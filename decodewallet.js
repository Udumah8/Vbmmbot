import fs from 'fs';
import { Keypair } from '@solana/web3.js';

//const { Keypair } = require('@solana/web3.js');
const walletData = JSON.parse(fs.readFileSync('config/wallets.json', 'utf8'));
const privateKey = walletData[0].privateKey;
const decrypted = require('./vbmbotcore.js').prototype.decrypt(privateKey); // Adjust path if needed
const keypair = Keypair.fromSecretKey(Buffer.from(decrypted, 'base64'));
console.log('Public Key:', keypair.publicKey.toBase58());