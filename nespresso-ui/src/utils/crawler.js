const axios = require('axios');
const cheerio = require('cheerio');

async function fetchHTML(url) {
    try {
        const { data } = await axios.get(url);
        return data;
    } catch (error) {
        console.error(`Error fetching ${url}: ${error}`);
        return null;
    }
}

function extractLinks(html, baseUrl) {
    const $ = cheerio.load(html);
    const links = [];
    $('a').each((i, link) => {
        const href = $(link).attr('href');
        if (href && href.startsWith('http')) {
            links.push(href);
        } else if (href) { // Handling relative URLs
            const newUrl = new URL(href, baseUrl).href;
            links.push(newUrl);
        }
    });
    return links;
}

async function crawl(url, maxDepth = 3) {
    const visited = new Set();
    const toVisit = [{ url, depth: 0 }];

    while (toVisit.length) {
        const { url, depth } = toVisit.shift();
        if (visited.has(url) || depth > maxDepth) continue;
        console.log(`Crawling ${url}`);
        visited.add(url);
        const html = await fetchHTML(url);
        if (!html) continue;
        const links = extractLinks(html, url);
        links.forEach(link => {
            if (!visited.has(link)) {
                toVisit.push({ url: link, depth: depth + 1 });
            }
        });
    }
}

// Usage
const startUrl = 'https://dining.fsu.edu/'; // Start from your target URL
crawl(startUrl, 2); // Set maxDepth as needed