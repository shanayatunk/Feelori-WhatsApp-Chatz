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

// Base customer type used in lists
export interface Customer {
  _id: string;
  phone_number: string;
  name: string;
  last_interaction: string;
}

// THIS IS THE CORRECTED TYPE for the detailed customer view
export interface CustomerDetails extends Customer {
    conversation_history: {
        timestamp: string;
        message: string;
        response: string;
    }[];
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

  return fetch(url, { ...options, headers });
};


export const apiService = {
  login: async (password: string): Promise<{ access_token: string }> => {
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

  getDashboardStats: async (): Promise<StatsData> => {
      const response = await makeRequest(`${API_BASE_URL}/dashboard/stats`);
      if (!response.ok) throw new Error('Failed to fetch stats');
      const result = await response.json();
      return result.data;
  },

  getHealth: async (): Promise<HealthData> => {
    const response = await makeRequest(`${API_BASE_URL}/admin/health`);
    if (!response.ok) throw new Error('Failed to fetch health status');
    return response.json();
  },

  getSecurityEvents: async (): Promise<SecurityEvent[]> => {
    const response = await makeRequest(`${API_BASE_URL}/admin/security-events`);
    if (!response.ok) throw new Error('Failed to fetch security events');
    const result = await response.json();
    return result.data.events;
  },

  getCustomers: async (page: number, limit: number): Promise<{ customers: Customer[], pagination: Pagination }> => {
    const response = await makeRequest(`${API_BASE_URL}/admin/customers?page=${page}&limit=${limit}`);
    if (!response.ok) throw new Error('Failed to fetch customers');
    const result = await response.json();
    return result.data;
  },

  // This function now correctly returns a promise of the corrected CustomerDetails type
  getCustomerById: async (id: string): Promise<{ customer: CustomerDetails }> => {
    const response = await makeRequest(`${API_BASE_URL}/admin/customers/${id}`);
    if (!response.ok) throw new Error('Failed to fetch customer details');
    const result = await response.json();
    return result.data;
  },

  getEscalations: async (): Promise<Escalation[]> => {
    const response = await makeRequest(`${API_BASE_URL}/dashboard/escalations`);
    if (!response.ok) throw new Error('Failed to fetch escalations');
    const result = await response.json();
    return result.data;
  },

  broadcast: async (message: string, target: string, imageUrl?: string): Promise<{ message: string }> => {
    const response = await makeRequest(`${API_BASE_URL}/admin/broadcast`, {
      method: 'POST',
      body: JSON.stringify({ message, target, image_url: imageUrl }),
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Broadcast failed' }));
        throw new Error(error.detail || 'Broadcast failed');
    }
    return response.json();
  },

  getPackingMetrics: async (): Promise<PackingMetrics> => {
      const response = await makeRequest(`${API_BASE_URL}/dashboard/packing-metrics`);
      if (!response.ok) throw new Error('Failed to fetch packing metrics');
      const result = await response.json();
      return result.data;
  },

  getRules: async (): Promise<Rule[]> => {
    const response = await makeRequest(`${API_BASE_URL}/admin/rules`);
    if (!response.ok) throw new Error('Failed to fetch rules');
    const result = await response.json();
    return result.data.rules;
  },

  createRule: async (rule: Rule): Promise<Rule> => {
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

  updateRule: async (ruleId: string, rule: Rule): Promise<Rule> => {
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

  getStrings: async (): Promise<StringResource[]> => {
    const response = await makeRequest(`${API_BASE_URL}/admin/strings`);
    if (!response.ok) {
      throw new Error(`Failed to fetch strings: ${response.status} ${response.statusText}`);
    }
    const result = await response.json();
    return result.data.strings;
  },

  updateStrings: async (strings: StringResource[]): Promise<void> => {
    const response = await makeRequest(`${API_BASE_URL}/admin/strings`, {
      method: 'PUT',
      body: JSON.stringify({ strings }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to update strings' }));
      throw new Error(error.detail || 'Failed to update strings');
    }
  },
};
