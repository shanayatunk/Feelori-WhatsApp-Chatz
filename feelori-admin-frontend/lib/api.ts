const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.feelori.com/api/v1';

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

// Add Type for Broadcast Group
export interface BroadcastGroup {
  id: string; // Corresponds to _id from backend
  name: string;
  phone_count: number;
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

// ADDED from previous step if missing
export const resolveTriageTicket = async (ticketId: string): Promise<{ message: string }> => {
  return makeRequest(`${API_BASE_URL}/triage/${ticketId}/resolve`, {
    method: 'PUT',
  });
};

// ADDED from previous step if missing
export const getTriageMediaUrl = (mediaId: string): string => {
   const token = typeof window !== 'undefined' ? localStorage.getItem('feelori_admin_token') : null;
   // Assuming the media endpoint is directly under /triage and needs auth
   return `${API_BASE_URL}/triage/media/${mediaId}`; // Token will be added by makeRequest header
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

  // --- NEW Broadcast Group Functions ---
  createBroadcastGroup: async (name: string, phoneNumbers: string[]): Promise<{ group_id: string }> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/broadcast-groups`, {
      method: 'POST',
      body: JSON.stringify({ name, phone_numbers: phoneNumbers }),
    });
    // Ensure the backend response structure matches if needed, e.g., result.data.group_id
    return result.data;
  },

  getBroadcastGroups: async (): Promise<BroadcastGroup[]> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/broadcast-groups`);
    // Map backend _id to frontend id
    return result.data.groups.map((g: any) => ({ ...g, id: g._id }));
  },
  // --- END NEW ---

  // --- UPDATED Broadcast Function ---
  broadcast: async (
      message: string,
      targetType: string,
      imageUrl?: string,
      targetPhones?: string[], // Kept for potential future use
      targetGroupId?: string   // Add group ID parameter
    ): Promise<{ message: string; job_id: string }> => { // Return job_id too

    let body: any = { message, target_type: targetType, image_url: imageUrl };

    if (targetType === 'custom_group' && targetGroupId) {
        body.target_group_id = targetGroupId;
    } else if (targetType === 'custom' && targetPhones) { // Renamed targetType for clarity if using phone list
        body.target_phones = targetPhones;
        // Backend expects 'custom' for phone list based on current structure, adjust if needed
    } else if (['all', 'active', 'inactive', 'recent'].includes(targetType)) {
        // No extra body params needed for standard types
    } else if (targetType !== 'custom_group') { // Avoid error if custom_group is selected but ID is missing initially
         throw new Error(`Invalid target type or missing required parameters for broadcast: ${targetType}`);
    }
    // If targetType is custom_group but targetGroupId is missing, the backend should handle validation

    const result = await makeRequest(`${API_BASE_URL}/admin/broadcast`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
    // Assuming backend returns job_id in data
    return { message: result.message, job_id: result.data?.job_id };
  },
  // --- END UPDATED ---

  getBroadcastRecipients: async (jobId: string, page: number, search: string): Promise<{ recipients: Recipient[], pagination: Pagination }> => {
    // Ensure search parameter is properly encoded if it contains special characters
    const searchParam = search ? `&search=${encodeURIComponent(search)}` : '';
    const result = await makeRequest(`${API_BASE_URL}/admin/broadcasts/${jobId}/recipients?page=${page}&limit=20${searchParam}`);
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

  getPackingMetrics: async (): Promise<PackingMetrics> => {
      // Assuming this should point to the performance endpoint now? Or a different one?
      // Using packer-performance as per previous context. Adjust if there's a specific /packing/metrics still.
      const result = await makeRequest(`${API_BASE_URL}/admin/packer-performance`); // Or /packing/metrics if that exists
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
      body: JSON.stringify({ strings }), // Ensure body structure matches backend Pydantic model
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
    const fullUrl = `${API_BASE_URL}/triage`;

    const result = await makeRequest(fullUrl);
    return result.data;
  } catch (error) {
    console.error("Failed to fetch triage tickets:", error);
    // Rethrow or return an empty structure
    throw new Error("Failed to fetch triage tickets");
    // return { tickets: [] }; // Alternative: return empty list on error
  }
};