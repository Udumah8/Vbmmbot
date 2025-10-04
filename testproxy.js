const axios = require('axios');

async function testProxy(proxy) {
  try {
    const [protocol, rest] = proxy.split('://');
    let host, port, username, password;
    if (rest.includes('@')) {
      [[username, password], [host, port]] = rest.split('@').map(part => part.split(':'));
    } else {
      [host, port] = rest.split(':');
    }
    const response = await axios.get('https://api.mainnet-beta.solana.com/health', {
      proxy: { host, port: parseInt(port), auth: username ? { username, password } : undefined },
      timeout: 10000
    });
    console.log(`Proxy ${proxy}: ${response.data}`);
  } catch (error) {
    console.error(`Proxy ${proxy} failed: ${error.message}`);
  }
}

testProxy('http://user123:pass123@proxy1.brightdata.com:8888');