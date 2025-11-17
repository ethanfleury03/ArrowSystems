/**
 * Frontend logging utility
 * Writes logs to a file on the server side via API route
 * Respects NEXT_PUBLIC_LOG_LEVEL environment variable
 */

import { getLogLevel, shouldLog, isDev } from './env';

export enum LogLevel {
  DEBUG = 'DEBUG',
  INFO = 'INFO',
  WARNING = 'WARNING',
  ERROR = 'ERROR',
  CRITICAL = 'CRITICAL',
}

interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  logger?: string;
  metadata?: Record<string, any>;
}

class Logger {
  private loggerName: string;
  private logBuffer: LogEntry[] = [];
  private flushInterval: NodeJS.Timeout | null = null;
  private readonly bufferSize = 10;
  private readonly flushIntervalMs = 5000; // 5 seconds

  constructor(loggerName: string = 'frontend') {
    this.loggerName = loggerName;
    this.startAutoFlush();
  }

  private formatMessage(level: LogLevel, message: string, metadata?: Record<string, any>): LogEntry {
    return {
      timestamp: new Date().toISOString(),
      level,
      message,
      logger: this.loggerName,
      metadata,
    };
  }

  private async writeLog(entry: LogEntry): Promise<void> {
    // Add to buffer
    this.logBuffer.push(entry);

    // Flush if buffer is full
    if (this.logBuffer.length >= this.bufferSize) {
      await this.flush();
    }
  }

  private async flush(): Promise<void> {
    if (this.logBuffer.length === 0) return;

    const logsToFlush = [...this.logBuffer];
    this.logBuffer = [];

    try {
      // Send logs to API route which writes to file
      await fetch('/api/logs/write', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ logs: logsToFlush }),
      }).catch((error) => {
        // Silently fail - don't break the app if logging fails
        console.error('Failed to write logs:', error);
      });
    } catch (error) {
      // Silently fail
      console.error('Failed to flush logs:', error);
    }
  }

  private startAutoFlush(): void {
    // Flush logs periodically
    if (typeof window !== 'undefined') {
      this.flushInterval = setInterval(() => {
        this.flush();
      }, this.flushIntervalMs);
    }
  }

  private stopAutoFlush(): void {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
      this.flushInterval = null;
    }
  }

  async debug(message: string, metadata?: Record<string, any>): Promise<void> {
    if (!shouldLog('debug')) {
      return;
    }
    const entry = this.formatMessage(LogLevel.DEBUG, message, metadata);
    // Only log to console in dev mode
    if (isDev) {
      console.debug(`[${this.loggerName}] ${message}`, metadata || '');
    }
    await this.writeLog(entry);
  }

  async info(message: string, metadata?: Record<string, any>): Promise<void> {
    if (!shouldLog('info')) {
      return;
    }
    const entry = this.formatMessage(LogLevel.INFO, message, metadata);
    // Only log to console in dev mode
    if (isDev) {
      console.info(`[${this.loggerName}] ${message}`, metadata || '');
    }
    await this.writeLog(entry);
  }

  async warning(message: string, metadata?: Record<string, any>): Promise<void> {
    if (!shouldLog('warn')) {
      return;
    }
    const entry = this.formatMessage(LogLevel.WARNING, message, metadata);
    // Always log warnings to console
    console.warn(`[${this.loggerName}] ${message}`, metadata || '');
    await this.writeLog(entry);
  }

  async error(message: string, error?: Error | any, metadata?: Record<string, any>): Promise<void> {
    if (!shouldLog('error')) {
      return;
    }
    const errorMessage = error instanceof Error ? `${message}: ${error.message}` : message;
    const errorMetadata = {
      ...metadata,
      error: error instanceof Error ? {
        name: error.name,
        message: error.message,
        stack: error.stack,
      } : error,
    };
    const entry = this.formatMessage(LogLevel.ERROR, errorMessage, errorMetadata);
    // Always log errors to console
    console.error(`[${this.loggerName}] ${errorMessage}`, errorMetadata);
    await this.writeLog(entry);
  }

  async critical(message: string, error?: Error | any, metadata?: Record<string, any>): Promise<void> {
    // Critical errors are always logged
    const errorMessage = error instanceof Error ? `${message}: ${error.message}` : message;
    const errorMetadata = {
      ...metadata,
      error: error instanceof Error ? {
        name: error.name,
        message: error.message,
        stack: error.stack,
      } : error,
    };
    const entry = this.formatMessage(LogLevel.CRITICAL, errorMessage, errorMetadata);
    // Always log critical errors to console
    console.error(`[${this.loggerName}] [CRITICAL] ${errorMessage}`, errorMetadata);
    await this.writeLog(entry);
    // Flush immediately for critical errors
    await this.flush();
  }

  // Flush remaining logs (call this on page unload)
  async destroy(): Promise<void> {
    this.stopAutoFlush();
    await this.flush();
  }
}

// Global logger instance
export const logger = new Logger('frontend');

// Create logger instances for specific modules
export function createLogger(name: string): Logger {
  return new Logger(name);
}

// Flush logs on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    logger.destroy();
  });
}





