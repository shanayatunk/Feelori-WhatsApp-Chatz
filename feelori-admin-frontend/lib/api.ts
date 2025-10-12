const API_BASE_URL = 'https://api.feelori.com/api/v1';

console.log('API_BASE_URL being used:', API_BASE_URL);
console.log('Environment variable was:', process.env.NEXT_PUBLIC_API_URL);

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
  conversation_volume: { _id: string; count: number }[];
}

export interface Recipient {
  _id: string;
  phone: string;
  status: string;
  customer_info?: {
      name: string;
  };
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

export interface PackerPerformanceData {
  kpis: {
    total_orders: number;
    completed_orders: number;
    on_hold_orders: number;
    avg_time_to_pack_minutes: number;
    hold_rate: number;
  };
  packer_leaderboard: {
    _id: string; // Packer name
    orders_packed: number;
  }[];
  hold_analysis: {
    by_reason: {
      _id: string; // Reason
      count: number;
    }[];
    top_problem_skus: {
      _id: string; // SKU
      count: number;
    }[];
  };
}


export interface BroadcastJob {
  _id: string;
  created_at: string;
  message: string;
  target_type: string;
  status: string;
  stats: {
    total_recipients: number;
    sent: number;
    delivered: number;
    read: number;
    failed: number;
  };
}

export interface BroadcastDetails extends BroadcastJob {
  image_url?: string;
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

export const resolveTriageTicket = async (ticketId: string): Promise<{ message: string }> => {
  return makeRequest(`${API_BASE_URL}/triage/${ticketId}/resolve`, {
    method: 'PUT',
  });
};

export const getTriageMediaUrl = (mediaId: string): string => {
  return `${API_BASE_URL}/triage/media/${mediaId}`;
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

  getBroadcastRecipients: async (jobId: string, page: number, search: string): Promise<{ recipients: Recipient[], pagination: Pagination }> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/broadcasts/${jobId}/recipients?page=${page}&limit=20&search=${search}`);
    return result.data;
  },


  getHealth: async (): Promise<HealthData> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/health`);
    return result.data;
  },

  getBroadcasts: async (page: number, limit: number): Promise<{ broadcasts: BroadcastJob[], pagination: Pagination }> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/broadcasts?page=${page}&limit=${limit}`);
    return result.data;
  },

  getBroadcastDetails: async (jobId: string): Promise<BroadcastDetails> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/broadcasts/${jobId}`);
    return result.data.details;
  },


  getPackerPerformance: async (days: number): Promise<PackerPerformanceData> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/packer-performance?days=${days}`);
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
    const result = await makeRequest(`${API_BASE_URL}/admin/escalations`);
    return result.data.escalations;
  },

  broadcast: async (message: string, target: string, imageUrl?: string): Promise<{ message: string }> => {
    return makeRequest(`${API_BASE_URL}/admin/broadcast`, {
      method: 'POST',
      body: JSON.stringify({ message, target_type: target, image_url: imageUrl }),
    });
  },

  getPackingMetrics: async (): Promise<PackingMetrics> => {
      const result = await makeRequest(`${API_BASE_URL}/admin/packer-performance`);
      return result.data;
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

// ADD THIS NEW TYPE
export type TriageTicket = {
  _id: string;
  customer_phone: string;
  order_number: string;
  issue_type: string;
  image_media_id: string | null;
  status: 'pending' | 'resolved';
  created_at: string;
};

// ADD THIS NEW API FUNCTION
export const getTriageTickets = async (): Promise<{ tickets: TriageTicket[] }> => {
  try {
    // --- THIS IS THE FIX ---
    // The endpoint has been changed from '/dashboard/triage-tickets' to the correct '/triage'
    const result = await makeRequest(`${API_BASE_URL}/triage`);
    return result.data;
  } catch (error) {
    console.error("Failed to fetch triage tickets:", error);
    throw new Error("Failed to fetch triage tickets");
  }
};