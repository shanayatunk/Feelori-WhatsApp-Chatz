import { StatsData, HealthData, SecurityEvent, Customer } from '../app/page';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export type Rule = {
  _id?: string;
  name: string;
  keywords: string[];
  phrases: string[];
};

export type StringResource = {
    key: string;
    value: string;
};

type ApiService = {
  login: (password: string) => Promise<{ access_token: string }>;
  getStats: () => Promise<StatsData>;
  getHealth: () => Promise<HealthData>;
  getSecurityEvents: () => Promise<SecurityEvent[]>;
  getCustomers: (page: number, limit: number) => Promise<{ customers: Customer[], pagination: any }>;
  broadcast: (message: string, target: string, imageUrl?: string) => Promise<any>;
  getPackingMetrics: () => Promise<any>;
  getRules: () => Promise<Rule[]>;
  createRule: (rule: Rule) => Promise<Rule>;
  updateRule: (ruleId: string, rule: Rule) => Promise<Rule>;
  getStrings: () => Promise<StringResource[]>;
  updateStrings: (strings: StringResource[]) => Promise<void>;
};

const api = (options?: RequestInit): typeof fetch => {
  const token = typeof window !== 'undefined' ? localStorage.getItem('feelori_admin_token') : null;
  const headers = {
    'Content-Type': 'application/json',
    ...options?.headers,
  };

  if (token) {
    (headers as any)['Authorization'] = `Bearer ${token}`;
  }

  return (url, fetchOptions) => fetch(url, { ...options, ...fetchOptions, headers });
};

export const apiService: ApiService = {
  login: async (password) => {
    const response = await api().post(`${API_BASE_URL}/auth/login`, {
      json: { password },
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Login failed');
    }
    return response.json();
  },

  getStats: async () => {
    const response = await api().get(`${API_BASE_URL}/admin/stats`);
    if (!response.ok) throw new Error('Failed to fetch stats');
    const result = await response.json();
    return result.data;
  },

  getHealth: async () => {
    const response = await api().get(`${API_BASE_URL}/admin/health`);
    if (!response.ok) throw new Error('Failed to fetch health');
    const result = await response.json();
    return result.data;
  },

  getSecurityEvents: async () => {
    const response = await api().get(`${API_BASE_URL}/admin/security/events`);
    if (!response.ok) throw new Error('Failed to fetch security events');
    const result = await response.json();
    return result.data.events;
  },

  getCustomers: async (page, limit) => {
    const response = await api().get(`${API_BASE_URL}/admin/customers?page=${page}&limit=${limit}`);
    if (!response.ok) throw new Error('Failed to fetch customers');
    const result = await response.json();
    return result.data;
  },

  broadcast: async (message, target, imageUrl) => {
    const response = await api().post(`${API_BASE_URL}/admin/broadcast`, {
      json: { message, target_type: target, image_url: imageUrl },
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Broadcast failed');
    }
    return response.json();
  },

  getPackingMetrics: async () => {
    const response = await api().get(`${API_BASE_URL}/packing/metrics`);
    if (!response.ok) throw new Error('Failed to fetch packing metrics');
    const result = await response.json();
    return result.data;
  },

  getRules: async () => {
    const response = await api().get(`${API_BASE_URL}/admin/rules`);
    if (!response.ok) throw new Error('Failed to fetch rules');
    const result = await response.json();
    return result.data.rules;
  },

  createRule: async (rule) => {
    const response = await api().post(`${API_BASE_URL}/admin/rules`, { json: rule });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create rule');
    }
    const result = await response.json();
    return result.data.rule;
  },

  updateRule: async (ruleId, rule) => {
    const response = await api().put(`${API_BASE_URL}/admin/rules/${ruleId}`, { json: rule });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to update rule');
    }
    const result = await response.json();
    return result.data.rule;
  },

  getStrings: async () => {
    const response = await api().get(`${API_BASE_URL}/admin/strings`);
    if (!response.ok) throw new Error('Failed to fetch strings');
    const result = await response.json();
    return result.data.strings;
    },

    updateStrings: async (strings) => {
    const response = await api().put(`${API_BASE_URL}/admin/strings`, { json: strings });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to update strings');
    }
    },
};

// Add helper methods to the fetch function
(global as any).fetch = (url: string, options?: RequestInit) => {
  const method = options?.method || 'GET';
  const newOptions = { ...options };
  if (options?.json) {
    newOptions.body = JSON.stringify(options.json);
    delete newOptions.json;
  }
  return (global as any)._fetch(url, newOptions);
};

['get', 'post', 'put', 'delete', 'patch'].forEach(method => {
  (api as any)()[method] = (url: string, options = {}) => (api as any)()(url, { ...options, method: method.toUpperCase() });
});
