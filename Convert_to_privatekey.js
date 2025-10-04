import bs58 from 'bs58';
import fs from 'fs';
// install: npm install bs58

// Your base64 private key (example: wallet 0)
const base64Key = "tUWMppvaYejm7ykYodVV72irEva9W9hPd6Ko9F6ibUdzJntA8Vu2beaJWOUd87pB6bdSfKWnMnBGC4NGHkIrQA==";

// Decode base64 → buffer
const raw = Buffer.from(base64Key, "base64");

// Convert to Uint8Array
const arr = Array.from(raw);

// Save as JSON file
fs.writeFileSync("wallet0.json", JSON.stringify(arr));

console.log("✅ Saved wallet0.json, ready for Phantom import");
