// Replace 'YOUR_ACTUAL_PRIVATE_KEY_ARRAY' with your key
const secretKey = Uint8Array.from([254, 148, 237, 176, 15, 40, 36, 88, 146, 198, 114, 233, 193, 166, 1, 139, 111, 107, 187, 26, 41, 123, 49, 134, 113, 139, 121, 170, 190, 137, 122, 236, 234, 55, 180, 139, 224, 229, 128, 124, 238, 229, 216, 71, 79, 55, 141, 37, 71, 167, 38, 178, 36, 200, 158, 135, 232, 6, 65, 230, 92, 167, 95, 5]);
const base64Key = Buffer.from(secretKey).toString('base64');
console.log(base64Key);
