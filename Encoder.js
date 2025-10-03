// Replace 'YOUR_ACTUAL_PRIVATE_KEY_ARRAY' with your key
const secretKey = Uint8Array.from([YOUR_ACTUAL_PRIVATE_KEY_ARRAY]);
const base64Key = Buffer.from(secretKey).toString('base64');
console.log(base64Key);
