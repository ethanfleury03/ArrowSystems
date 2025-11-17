/** @type {import('next').NextConfig} */

// Build-time environment validation for production
const validateProductionEnv = () => {
  const isProduction = process.env.NODE_ENV === 'production';
  
  if (isProduction) {
    const errors = [];
    
    // Require NEXT_PUBLIC_API_URL in production
    if (!process.env.NEXT_PUBLIC_API_URL) {
      errors.push(
        'NEXT_PUBLIC_API_URL is required in production but was not set. ' +
        'Set this environment variable to your production API URL.'
      );
    }
    
    // Require SESSION_SECRET in production
    if (!process.env.SESSION_SECRET) {
      errors.push(
        'SESSION_SECRET is required in production but was not set. ' +
        'Set a secure random string of at least 32 characters.'
      );
    } else {
      // Validate SESSION_SECRET is not a default value
      const unsafeDefaults = [
        'change-this-to-a-random-string-at-least-32-characters-long',
        'dev-session-secret-not-for-production',
        'secret',
        'password',
        'default-secret',
      ];
      
      if (unsafeDefaults.includes(process.env.SESSION_SECRET)) {
        errors.push(
          'SESSION_SECRET is set to an unsafe default value. ' +
          'In production, use a secure random string of at least 32 characters.'
        );
      } else if (process.env.SESSION_SECRET.length < 32) {
        errors.push(
          'SESSION_SECRET is too short. ' +
          'In production, SESSION_SECRET must be at least 32 characters long.'
        );
      }
    }
    
    // Validate NEXT_PUBLIC_LOG_LEVEL if provided
    if (process.env.NEXT_PUBLIC_LOG_LEVEL) {
      const validLevels = ['debug', 'info', 'warn', 'error'];
      if (!validLevels.includes(process.env.NEXT_PUBLIC_LOG_LEVEL.toLowerCase())) {
        errors.push(
          `NEXT_PUBLIC_LOG_LEVEL must be one of: ${validLevels.join(', ')}. ` +
          `Got: ${process.env.NEXT_PUBLIC_LOG_LEVEL}`
        );
      }
    }
    
    if (errors.length > 0) {
      console.error('\n❌ Production build validation failed:\n');
      errors.forEach((error, index) => {
        console.error(`  ${index + 1}. ${error}`);
      });
      console.error('\n');
      throw new Error('Production build validation failed. See errors above.');
    }
    
    console.log('✅ Production build validation passed');
  }
};

// Run validation before building
validateProductionEnv();

const nextConfig = {
  reactStrictMode: true,
  output: 'standalone', // Required for Docker
  
  // Remove X-Powered-By header for security
  poweredByHeader: false,
  
  // Enable gzip compression
  compress: true,
  
  // Security headers
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on'
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN'
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff'
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin'
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()'
          }
        ]
      }
    ];
  }
};

module.exports = nextConfig;

