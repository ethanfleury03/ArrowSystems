'use client';

import { useEffect, useState } from 'react';
import { resolveApiBaseUrl, resolveInitialBaseUrl } from '@/config/api';

const LOCAL_KEY = 'useLocalBackend';

export function ApiToggle() {
  const [isLocal, setIsLocal] = useState(false);
  const [activeUrl, setActiveUrl] = useState(resolveInitialBaseUrl());

  useEffect(() => {
    const resolvedUrl = resolveApiBaseUrl();
    setActiveUrl(resolvedUrl);
    setIsLocal(localStorage.getItem(LOCAL_KEY) === 'true');
  }, []);

  const handleToggle = () => {
    setIsLocal((prev) => {
      const nextValue = !prev;
      if (nextValue) {
        localStorage.setItem(LOCAL_KEY, 'true');
      } else {
        localStorage.removeItem(LOCAL_KEY);
      }
      // Reload so modules pick up the new base URL
      window.location.reload();
      return nextValue;
    });
  };

  return (
    <div className="flex items-center gap-3 rounded border border-border bg-background px-3 py-2 text-sm">
      <div>
        <div className="font-medium">Backend base URL</div>
        <div className="text-muted-foreground break-all">{activeUrl}</div>
      </div>
      <button
        type="button"
        onClick={handleToggle}
        className="ml-auto rounded bg-primary px-3 py-1 text-primary-foreground"
      >
        {isLocal ? 'Use Remote' : 'Use Local'}
      </button>
    </div>
  );
}

