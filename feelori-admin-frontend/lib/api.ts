const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

// --- Type Definitions ---
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

export interface StatsData {
  total_customers: number;
  active_conversations: number;
  human_escalations: number;
  avg_response_time: string;
}

export interface HealthData {
  status: string;
  services: Record<string, string>;
}

export interface SecurityEvent {
  _id: string;
  event_type: string;
  ip_address: string;
  timestamp: string;
  details: Record<string, unknown>;
}

export interface Customer {
  _id: string;
  phone_number: string;
  name: string;
  last_interaction: string;
}

export interface CustomerDetails extends Customer {
    conversation_history: { sender: string; message: string; timestamp: string }[];
}

export interface Escalation {
  _id: string;
  name: string;
  phone_number: string;
}

export interface PackingMetrics {
    status_counts: Record<string, number>;
}

export interface Pagination {
    page: number;
    limit: number;
    total: number;
    pages: number;
}


// --- API Service ---
const makeRequest = async (url: string, options: RequestInit = {}) => {
  const token = typeof window !== 'undefined' ? localStorage.getItem('feelori_admin_token') : null;

  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
    ...(token && { Authorization: `Bearer ${token}` }),
  };

  const response = await fetch(url, { ...options, headers });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'An unknown error occurred' }));
    throw new Error(error.detail || `Request failed with status ${response.status}`);
  }
  return response.json();
};


export const apiService = {
  login: async (password: string): Promise<{ access_token: string }> => {
    return makeRequest(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      body: JSON.stringify({ password }),
    });
  },

  getDashboardStats: async (): Promise<StatsData> => {
      const result = await makeRequest(`${API_BASE_URL}/dashboard/stats`);
      return result.data.stats;
  },

  getHealth: async (): Promise<HealthData> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/health`);
    return result.data;
  },

  getSecurityEvents: async (): Promise<SecurityEvent[]> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/security-events`);
    return result.data.events;
  },

  getCustomers: async (page: number, limit: number): Promise<{ customers: Customer[], pagination: Pagination }> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/customers?page=${page}&limit=${limit}`);
    return result.data;
  },

  getCustomerById: async (id: string): Promise<{ customer: CustomerDetails }> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/customers/${id}`);
    return result.data;
  },

  getEscalations: async (): Promise<Escalation[]> => {
    const result = await makeRequest(`${API_BASE_URL}/dashboard/escalations`);
    return result.data.escalations;
  },

  broadcast: async (message: string, target: string, imageUrl?: string): Promise<{ message: string }> => {
    return makeRequest(`${API_BASE_URL}/admin/broadcast`, {
      method: 'POST',
      body: JSON.stringify({ message, target, image_url: imageUrl }),
    });
  },

  getPackingMetrics: async (): Promise<PackingMetrics> => {
      const result = await makeRequest(`${API_BASE_URL}/dashboard/packing-metrics`);
      return result.data.metrics;
  },

  getRules: async (): Promise<Rule[]> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/rules`);
    return result.data.rules;
  },

  createRule: async (rule: Rule): Promise<Rule> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/rules`, {
      method: 'POST',
      body: JSON.stringify(rule),
    });
    return result.data.rule;
  },

  updateRule: async (ruleId: string, rule: Rule): Promise<Rule> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/rules/${ruleId}`, {
      method: 'PUT',
      body: JSON.stringify(rule),
    });
    return result.data.rule;
  },

  getStrings: async (): Promise<StringResource[]> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/strings`);
    return result.data.strings;
  },

  updateStrings: async (strings: StringResource[]): Promise<{ message: string }> => {
    return makeRequest(`${API_BASE_URL}/admin/strings`, {
      method: 'PUT',
      body: JSON.stringify({ strings }),
    });
  },
};