// const fs = require('fs');
// const { Keypair } = require('@solana/web3.js');
// const walletData = JSON.parse(fs.readFileSync('D:/PythonProject/pythonProject/vbmm/config/wallets.json', 'utf8'));
// const privateKey = walletData[0].privateKey;
// const bot = require('./v2.js').default;
// const decrypted = new bot().decrypt(privateKey);
// const keypair = Keypair.fromSecretKey(Buffer.from(decrypted, 'base64'));
// console.log('Public Key:', keypair.publicKey.toBase58());

import fs from 'fs';
import { Keypair } from '@solana/web3.js';
import bot from './v2.js'

const walletData = JSON.parse(fs.readFileSync('D:/PythonProject/pythonProject/vbmm/config/wallets.json', 'utf8'));
//const bot = require('./v2.js').default;
walletData.forEach((wallet, index) => {
    const decrypted = new bot().decrypt(wallet.privateKey);
    const keypair = Keypair.fromSecretKey(Buffer.from(decrypted, 'base64'));
    console.log(`Wallet ${index} Public Key: ${keypair.publicKey.toBase58()}`);
});