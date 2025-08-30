// /lib/api.ts

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
  getCustomerById: (id: string) => Promise<{ customer: any }>;
  getEscalations: () => Promise<any[]>;
  broadcast: (message: string, target: string, imageUrl?: string) => Promise<any>;
  getPackingMetrics: () => Promise<any>;
  getRules: () => Promise<Rule[]>;
  createRule: (rule: Rule) => Promise<Rule>;
  updateRule: (ruleId: string, rule: Rule) => Promise<Rule>;
  getStrings: () => Promise<StringResource[]>;
  updateStrings: (strings: StringResource[]) => Promise<void>;
};

// Helper function to make authenticated requests
const makeRequest = async (url: string, options: RequestInit = {}): Promise<Response> => {
  const token = typeof window !== 'undefined' ? localStorage.getItem('feelori_admin_token') : null;
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  if (token) {
    (headers as any)['Authorization'] = `Bearer ${token}`;
  }

  return fetch(url, { ...options, headers });
};


export const apiService: ApiService = {
  login: async (password) => {
    const response = await makeRequest(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      body: JSON.stringify({ password }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Login failed' }));
      throw new Error(error.detail || 'Login failed');
    }
    return response.json();
  },
  
  getStats: async () => {
    const response = await makeRequest(`${API_BASE_URL}/admin/stats`);
    if (!response.ok) throw new Error('Failed to fetch stats');
    const result = await response.json();
    return result.data;
  },

  getHealth: async () => {
    const response = await makeRequest(`${API_BASE_URL}/admin/health`);
    if (!response.ok) throw new Error('Failed to fetch health');
    const result = await response.json();
    return result.data;
  },

  getSecurityEvents: async () => {
    const response = await makeRequest(`${API_BASE_URL}/admin/security/events`);
    if (!response.ok) throw new Error('Failed to fetch security events');
    const result = await response.json();
    return result.data.events;
  },

  getCustomers: async (page, limit) => {
    const response = await makeRequest(`${API_BASE_URL}/admin/customers?page=${page}&limit=${limit}`);
    if (!response.ok) throw new Error('Failed to fetch customers');
    const result = await response.json();
    return result.data;
  },

  getCustomerById: async (id) => {
    const response = await makeRequest(`${API_BASE_URL}/admin/customers/${id}`);
    if (!response.ok) throw new Error('Failed to fetch customer details');
    const result = await response.json();
    return result.data;
  },

  getEscalations: async () => {
    const response = await makeRequest(`${API_BASE_URL}/admin/escalations`);
    if (!response.ok) throw new Error('Failed to fetch escalation requests');
    const result = await response.json();
    return result.data.escalations;
  },

  broadcast: async (message, target, imageUrl) => {
    const response = await makeRequest(`${API_BASE_URL}/admin/broadcast`, {
      method: 'POST',
      body: JSON.stringify({ message, target_type: target, image_url: imageUrl }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Broadcast failed' }));
      throw new Error(error.detail || 'Broadcast failed');
    }
    return response.json();
  },

  getPackingMetrics: async () => {
    const response = await makeRequest(`${API_BASE_URL}/packing/metrics`);
    if (!response.ok) throw new Error('Failed to fetch packing metrics');
    const result = await response.json();
    return result.data;
  },

  getRules: async () => {
    const response = await makeRequest(`${API_BASE_URL}/admin/rules`);
    if (!response.ok) throw new Error('Failed to fetch rules');
    const result = await response.json();
    return result.data.rules;
  },

  createRule: async (rule) => {
    const response = await makeRequest(`${API_BASE_URL}/admin/rules`, {
      method: 'POST',
      body: JSON.stringify(rule),
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to create rule' }));
      throw new Error(error.detail || 'Failed to create rule');
    }
    
    const result = await response.json();
    return result.data.rule;
  },

  updateRule: async (ruleId, rule) => {
    const response = await makeRequest(`${API_BASE_URL}/admin/rules/${ruleId}`, {
      method: 'PUT',
      body: JSON.stringify(rule),
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to update rule' }));
      throw new Error(error.detail || 'Failed to update rule');
    }
    
    const result = await response.json();
    return result.data.rule;
  },

  getStrings: async () => {
    const response = await makeRequest(`${API_BASE_URL}/admin/strings`);
    if (!response.ok) {
      throw new Error(`Failed to fetch strings: ${response.status} ${response.statusText}`);
    }
    const result = await response.json();
    return result.data.strings;
  },

  updateStrings: async (strings) => {
    const response = await makeRequest(`${API_BASE_URL}/admin/strings`, {
      method: 'PUT',
      body: JSON.stringify(strings),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to update strings' }));
      throw new Error(error.detail || 'Failed to update strings');
    }
  },
};