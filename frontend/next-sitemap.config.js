/** @type {import('next-sitemap').IConfig} */
module.exports = {
  siteUrl: process.env.NEXT_PUBLIC_SITE_URL || 'https://<DOMAIN>',
  generateRobotsTxt: true,
  sitemapSize: 7000,
  exclude: [
    '/api/*',
    '/admin',
    '/admin/*',
    '/login',
    '/register',
    '/accept-invite',
    '/account',
  ],
  // Ensure URLs are generated correctly (no trailing slash by default)
  trailingSlash: false,
}

