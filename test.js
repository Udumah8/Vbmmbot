import pkg from '@jup-ag/api';
import { Connection } from '@solana/web3.js';
const { Jupiter } = pkg;
async function test() {
  const connection = new Connection('https://api.mainnet-beta.solana.com');
  const jupiter = await Jupiter.load({ connection, cluster: 'mainnet-beta' });
  console.log('Jupiter loaded:', !!jupiter);
}
test();