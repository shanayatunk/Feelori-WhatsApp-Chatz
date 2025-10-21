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

// Define the type for the raw group data coming from the backend
interface RawBroadcastGroup {
  _id: string;
  name: string;
  phone_count: number;
}

// Define the type for the broadcast request body
interface BroadcastRequestBody {
    message: string;
    target_type: string;
    image_url?: string;
    target_group_id?: string;
    target_phones?: string[];
}


// --- API Service ---
const makeRequest = async (url: string, options: RequestInit = {}) => {

  // Get token only if window is defined (client-side)
  const token = typeof window !== 'undefined' ? localStorage.getItem('feelori_admin_token') : null;

  const headers: HeadersInit = { // Use HeadersInit type
    'Content-Type': 'application/json',
    ...options.headers,
  };

  // Add Authorization header if token exists
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(url, { ...options, headers });

  if (!response.ok) {
    let errorDetail = 'An unknown error occurred';
    try {
        const error = await response.json();
        errorDetail = error.detail || `Request failed with status ${response.status}`;
    } catch (e) {
        // If parsing error response fails, use status text
        errorDetail = `Request failed with status ${response.status}: ${response.statusText}`;
    }
    throw new Error(errorDetail);
  }

  // Handle cases where response might be empty (e.g., 204 No Content)
  if (response.status === 204) {
      return null; // Or return an empty object/specific indicator if needed
  }

  return response.json();
};

export const resolveTriageTicket = async (ticketId: string): Promise<{ message: string }> => {
  return makeRequest(`${API_BASE_URL}/triage/${ticketId}/resolve`, {
    method: 'PUT',
  });
};

// FIX: Removed token logic - makeRequest handles Authorization header
export const getTriageMediaUrl = (mediaId: string): string => {
   // Just return the URL, the caller should handle fetching with authentication
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
      if (!result || !result.data || !result.data.stats) {
          throw new Error("Invalid stats data received from API");
      }
      return result.data.stats;
  },

  // --- NEW Broadcast Group Functions ---
  createBroadcastGroup: async (name: string, phoneNumbers: string[]): Promise<{ group_id: string }> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/broadcast-groups`, {
      method: 'POST',
      body: JSON.stringify({ name, phone_numbers: phoneNumbers }),
    });
    if (!result || !result.data || !result.data.group_id) {
        throw new Error("Invalid response when creating broadcast group");
    }
    return result.data;
  },

  getBroadcastGroups: async (): Promise<BroadcastGroup[]> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/broadcast-groups`);
    if (!result || !result.data || !Array.isArray(result.data.groups)) {
         throw new Error("Invalid groups data received from API");
    }
    // FIX: Use defined RawBroadcastGroup type for mapping
    return result.data.groups.map((g: RawBroadcastGroup) => ({
        id: g._id, // Map _id to id
        name: g.name,
        phone_count: g.phone_count
    }));
  },
  // --- END NEW ---

  // --- UPDATED Broadcast Function ---
  broadcast: async (
      message: string,
      targetType: string,
      imageUrl?: string,
      targetPhones?: string[],
      targetGroupId?: string
    ): Promise<{ message: string; job_id: string }> => {

    // FIX: Use const for body and define its type
    const body: BroadcastRequestBody = { message, target_type: targetType };

    if (imageUrl) {
        body.image_url = imageUrl;
    }

    if (targetType === 'custom_group' && targetGroupId) {
        body.target_group_id = targetGroupId;
    } else if (targetType === 'custom' && targetPhones) { // Using 'custom' if backend expects it for phone list
        body.target_phones = targetPhones;
    } else if (!['all', 'active', 'inactive', 'recent', 'custom_group'].includes(targetType)) {
         // Throw error earlier for invalid types (unless custom_group is selected without ID yet)
         throw new Error(`Invalid target type or missing required parameters for broadcast: ${targetType}`);
    }

    const result = await makeRequest(`${API_BASE_URL}/admin/broadcast`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
    if (!result || !result.data) {
        throw new Error("Invalid response when sending broadcast");
    }
    return { message: result.message || "Broadcast initiated.", job_id: result.data.job_id };
  },
  // --- END UPDATED ---

  getBroadcastRecipients: async (jobId: string, page: number, search: string): Promise<{ recipients: Recipient[], pagination: Pagination }> => {
    const searchParam = search ? `&search=${encodeURIComponent(search)}` : '';
    const result = await makeRequest(`${API_BASE_URL}/admin/broadcasts/${jobId}/recipients?page=${page}&limit=20${searchParam}`);
     if (!result || !result.data) {
        throw new Error("Invalid recipients data received from API");
    }
    return result.data;
  },


  getHealth: async (): Promise<HealthData> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/health`);
    if (!result || !result.data) {
        throw new Error("Invalid health data received from API");
    }
    return result.data;
  },

  getBroadcasts: async (page: number, limit: number): Promise<{ broadcasts: BroadcastJob[], pagination: Pagination }> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/broadcasts?page=${page}&limit=${limit}`);
     if (!result || !result.data) {
        throw new Error("Invalid broadcasts data received from API");
    }
    return result.data;
  },

  getBroadcastDetails: async (jobId: string): Promise<BroadcastDetails> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/broadcasts/${jobId}`);
     if (!result || !result.data || !result.data.details) {
        throw new Error("Invalid broadcast details received from API");
    }
    return result.data.details;
  },


  getPackerPerformance: async (days: number): Promise<PackerPerformanceData> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/packer-performance?days=${days}`);
     if (!result || !result.data) {
        throw new Error("Invalid performance data received from API");
    }
    return result.data;
  },


  getSecurityEvents: async (): Promise<SecurityEvent[]> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/security-events`);
     if (!result || !result.data || !Array.isArray(result.data.events)) {
        throw new Error("Invalid security events data received from API");
    }
    return result.data.events;
  },

  getCustomers: async (page: number, limit: number): Promise<{ customers: Customer[], pagination: Pagination }> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/customers?page=${page}&limit=${limit}`);
     if (!result || !result.data) {
        throw new Error("Invalid customers data received from API");
    }
    return result.data;
  },

  getCustomerById: async (id: string): Promise<{ customer: CustomerDetails }> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/customers/${id}`);
     if (!result || !result.data || !result.data.customer) {
        throw new Error("Invalid customer details received from API");
    }
    return result.data;
  },

  getEscalations: async (): Promise<Escalation[]> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/escalations`);
     if (!result || !result.data || !Array.isArray(result.data.escalations)) {
        throw new Error("Invalid escalations data received from API");
    }
    return result.data.escalations;
  },

  getPackingMetrics: async (): Promise<PackingMetrics> => {
      const result = await makeRequest(`${API_BASE_URL}/admin/packer-performance`); // Assuming this endpoint provides necessary metrics
       if (!result || !result.data) {
           throw new Error("Invalid packing metrics data received from API");
       }
      // Adapt if the structure is different from PackerPerformanceData
      return result.data.kpis ? { status_counts: { // Example adaptation
          'Pending': result.data.kpis.total_orders - (result.data.kpis.completed_orders + result.data.kpis.on_hold_orders), // Approximate pending
          'On Hold': result.data.kpis.on_hold_orders,
          'Completed': result.data.kpis.completed_orders
          // Add 'In Progress' if available
        }} : { status_counts: {} };
  },

  getRules: async (): Promise<Rule[]> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/rules`);
     if (!result || !result.data || !Array.isArray(result.data.rules)) {
        throw new Error("Invalid rules data received from API");
    }
    return result.data.rules;
  },

  createRule: async (rule: Rule): Promise<Rule> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/rules`, {
      method: 'POST',
      body: JSON.stringify(rule),
    });
     if (!result || !result.data || !result.data.rule) {
        throw new Error("Invalid response when creating rule");
    }
    return result.data.rule;
  },

  updateRule: async (ruleId: string, rule: Rule): Promise<Rule> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/rules/${ruleId}`, {
      method: 'PUT',
      body: JSON.stringify(rule),
    });
     if (!result || !result.data || !result.data.rule) {
        throw new Error("Invalid response when updating rule");
    }
    return result.data.rule;
  },

  getStrings: async (): Promise<StringResource[]> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/strings`);
     if (!result || !result.data || !Array.isArray(result.data.strings)) {
        throw new Error("Invalid strings data received from API");
    }
    return result.data.strings;
  },

  updateStrings: async (strings: StringResource[]): Promise<{ message: string }> => {
    const result = await makeRequest(`${API_BASE_URL}/admin/strings`, {
      method: 'PUT',
      body: JSON.stringify({ strings }), // Ensure body structure matches backend Pydantic model
    });
    if (!result) {
        throw new Error("Invalid response when updating strings");
    }
    return { message: result.message || "Strings updated." };
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
    if (!result || !result.data || !Array.isArray(result.data.tickets)) {
        console.error("Invalid triage tickets data structure received:", result);
        throw new Error("Invalid triage tickets data received from API");
    }
    return result.data;
  } catch (error) {
    console.error("Failed to fetch triage tickets:", error);
    // Rethrow or return an empty structure
    throw new Error(`Failed to fetch triage tickets: ${error instanceof Error ? error.message : 'Unknown error'}`);
    // return { tickets: [] }; // Alternative: return empty list on error
  }
};