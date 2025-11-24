/**
 * Environment detection and configuration helpers
 * Provides dev/prod awareness for the frontend
 */

export const isDev = process.env.NODE_ENV === 'development';
export const isProd = process.env.NODE_ENV === 'production';

/**
 * Get log level from environment variable
 * Dev defaults to 'debug', Prod defaults to 'error'
 */
export const getLogLevel = (): 'debug' | 'info' | 'warn' | 'error' => {
  const envLevel = process.env.NEXT_PUBLIC_LOG_LEVEL?.toLowerCase();
  if (envLevel === 'debug' || envLevel === 'info' || envLevel === 'warn' || envLevel === 'error') {
    return envLevel;
  }
  return isDev ? 'debug' : 'error';
};

/**
 * Check if a log level should be logged based on current log level
 */
export const shouldLog = (level: 'debug' | 'info' | 'warn' | 'error'): boolean => {
  const currentLevel = getLogLevel();
  const levels: Record<string, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
  };
  return levels[level] >= levels[currentLevel];
};









