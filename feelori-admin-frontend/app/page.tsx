"use client";
import React from 'react';

// NOTE: This is a complete, self-contained Next.js application.
// All components are defined in a single file for clarity.

// --- 1. MOCK DATA & TYPES (lib/types.ts) ---
export type StatsData = {
  customers: { total: number; active_24h: number };
  messages: { total_24h: number };
  system: { queue_size: number };
};

export type HealthData = {
  status: string;
  services: {
    database: 'connected' | 'error';
    cache: 'connected' | 'error';
    whatsapp: 'configured' | 'not_configured';
    shopify: 'configured' | 'not_configured';
  };
};

export type SecurityEvent = {
  _id: string;
  event_type: string;
  ip_address: string;
  timestamp: string;
  details: Record<string, any>;
};

export type Customer = {
  _id: string;
  phone_number: string;
  name?: string;
  last_interaction: string;
  total_messages: number;
};

// --- 2. CORE UI COMPONENTS (components/ui/) ---
const Card = ({ children, className = '' }) => (
  <div className={`bg-white border border-gray-200 rounded-lg shadow-sm ${className}`}>
    {children}
  </div>
);

const CardHeader = ({ children, className = '' }) => (
  <div className={`p-4 sm:p-6 border-b border-gray-200 ${className}`}>
    {children}
  </div>
);

const CardContent = ({ children, className = '' }) => (
  <div className={`p-4 sm:p-6 ${className}`}>
    {children}
  </div>
);

const CardTitle = ({ children, className = '' }) => (
  <h3 className={`text-lg font-semibold text-black ${className}`}>{children}</h3>
);

const CardDescription = ({ children, className = '' }) => (
  <p className={`text-sm text-gray-700 ${className}`}>{children}</p>
);

const Button = ({ children, onClick, className = '', disabled = false, variant = 'primary' }) => {
  const baseClasses = `inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none px-4 py-2`;
  
  const variantClasses = {
    primary: 'bg-[#ff4d6d] text-white hover:bg-[#e6395b]',
    secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200 border border-gray-200',
    ghost: 'hover:bg-gray-100 hover:text-gray-900',
    success: 'bg-green-600 text-white hover:bg-green-700',
  };
  
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
    >
      {children}
    </button>
  );
};

const Input = ({ className = '', ...props }) => (
  <input
    className={`flex h-10 w-full rounded-md border border-gray-300 bg-transparent px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-[#ff4d6d] focus:ring-offset-2 ${className}`}
    {...props}
  />
);

const Textarea = ({ className = '', ...props }) => (
    <textarea
        className={`w-full rounded-md border border-gray-300 p-2 font-mono text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-[#ff4d6d] focus:ring-offset-2 ${className}`}
        {...props}
    />
);

const TagInput = ({ tags, setTags, placeholder }) => {
    const [inputValue, setInputValue] = React.useState('');

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && inputValue.trim()) {
            e.preventDefault();
            if (!tags.includes(inputValue.trim().toLowerCase())) {
                setTags([...tags, inputValue.trim().toLowerCase()]);
            }
            setInputValue('');
        }
    };

    const removeTag = (tagToRemove) => {
        setTags(tags.filter(tag => tag !== tagToRemove));
    };

    return (
        <div className="flex flex-wrap items-center gap-2 p-2 border border-gray-300 rounded-md bg-white">
            {tags.map(tag => (
                <span key={tag} className="flex items-center gap-1 bg-[#ff4d6d] text-white text-sm font-medium px-2 py-1 rounded-full">
                    {tag}
                    <button onClick={() => removeTag(tag)} className="text-white hover:text-gray-200">&times;</button>
                </span>
            ))}
            <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={placeholder}
                className="flex-grow bg-transparent outline-none text-sm text-gray-900"
            />
        </div>
    );
};


// --- 3. ICONS (components/icons.tsx) ---
const HomeIcon = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><polyline points="9 22 9 12 15 12 15 22" /></svg>
);
const UsersIcon = (props) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
);
const MessageSquareIcon = (props) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
);
const HeartPulseIcon = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" /><path d="M3.22 12H9.5l.7-1.5L13.5 14l.7-1.5H20.78" /></svg>
);
const ShieldCheckIcon = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" /><path d="m9 12 2 2 4-4" /></svg>
);
const SendIcon = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" /></svg>
);
const SettingsIcon = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 0 2l-.15.08a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.38a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1 0-2l.15-.08a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" /><circle cx="12" cy="12" r="3" /></svg>
);
const LogOutIcon = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" x2="9" y1="12" y2="12" /></svg>
);
const FileTextIcon = (props) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>
);
const PaperclipIcon = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.59a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
);


// --- 4. SHARED LAYOUT COMPONENTS (components/dashboard/) ---

const Sidebar = ({ activePage, setActivePage }) => {
  const navItems = [
    { type: 'header', label: 'Analytics' },
    { id: 'dashboard', label: 'Dashboard', icon: <HomeIcon className="h-5 w-5" /> },
    { id: 'conversations', label: 'Conversations', icon: <MessageSquareIcon className="h-5 w-5" /> },
    { id: 'performance', label: 'Packer Performance', icon: <UsersIcon className="h-5 w-5" /> },
    { type: 'divider' },
    { type: 'header', label: 'Administration' },
    { id: 'health', label: 'System Health', icon: <HeartPulseIcon className="h-5 w-5" /> },
    { id: 'security', label: 'Security', icon: <ShieldCheckIcon className="h-5 w-5" /> },
    { id: 'broadcast', label: 'Broadcast', icon: <SendIcon className="h-5 w-5" /> },
    { type: 'divider' },
    { type: 'header', label: 'Configuration' },
    { id: 'rules', label: 'Rules Engine', icon: <SettingsIcon className="h-5 w-5" /> },
    { id: 'strings', label: 'Strings Manager', icon: <FileTextIcon className="h-5 w-5" /> },
  ];

  return (
    <aside className="hidden w-64 flex-col border-r bg-gray-50 lg:flex">
      <div className="flex h-16 items-center border-b px-6">
        <a href="#" className="flex items-center gap-2 font-semibold">
          <span className="text-black">FeelOri Admin</span>
        </a>
      </div>
      <nav className="flex-1 overflow-auto py-4">
        <ul className="grid items-start px-4 text-sm font-medium">
          {navItems.map((item, index) => {
            if (item.type === 'header') {
                return <li key={index} className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">{item.label}</li>
            }
            if (item.type === 'divider') {
                return <li key={index}><hr className="my-2 border-gray-200" /></li>
            }
            return (
            <li key={item.id}>
              <a
                href="#"
                onClick={(e) => { e.preventDefault(); setActivePage(item.id); }}
                className={`flex items-center gap-3 rounded-lg px-3 py-2 transition-all ${activePage === item.id ? 'bg-[#ff4d6d] text-white' : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'}`}
              >
                {item.icon}
                {item.label}
              </a>
            </li>
            );
        })}
        </ul>
      </nav>
    </aside>
  );
};

const Header = ({ onLogout }) => (
  <header className="flex h-14 items-center gap-4 border-b bg-white px-6 lg:h-[60px]">
    <div className="w-full flex-1">
      {/* Can add a global search bar here in the future */}
    </div>
    <Button onClick={onLogout} variant="secondary">
      <LogOutIcon className="h-5 w-5 mr-2" /> Logout
    </Button>
  </header>
);

const DashboardLayout = ({ children, activePage, setActivePage, onLogout }) => (
  <div className="grid min-h-screen w-full lg:grid-cols-[256px_1fr]">
    <Sidebar activePage={activePage} setActivePage={setActivePage} />
    <div className="flex flex-col">
      <Header onLogout={onLogout} />
      <main className="flex-1 bg-gray-100 p-4 sm:p-6">{children}</main>
    </div>
  </div>
);


// --- 5. PAGE COMPONENTS (app/dashboard/*) ---

const StatCard = ({ title, value, icon, color = '#ff4d6d' }) => (
  <Card>
    <CardHeader className="flex flex-row items-center justify-between pb-2">
      <h3 className="text-sm font-medium text-gray-700">{title}</h3>
      {icon}
    </CardHeader>
    <CardContent>
      <div className="text-2xl font-bold" style={{ color }}>{value}</div>
    </CardContent>
  </Card>
);

const DashboardPage = ({ setPage }) => {
  const [stats, setStats] = React.useState(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    setTimeout(() => {
      setStats({
        customers: { total: 1256, active_24h: 89 },
        messages: { total_24h: 432 },
        system: { queue_size: 3 },
        support_requests: 14,
      });
      setLoading(false);
    }, 1000);
  }, []);

  if (loading) return <div className="text-center p-10 text-gray-900">Loading dashboard...</div>;

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-black">Business Performance</h1>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard title="Total Customers" value={stats?.customers.total ?? '...'} />
        <StatCard title="Active Customers (24h)" value={stats?.customers.active_24h ?? '...'} />
        <StatCard title="Messages (24h)" value={stats?.messages.total_24h ?? '...'} />
        <StatCard title="Human Support Requests" value={stats?.support_requests ?? '...'} />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
            <CardHeader><CardTitle>Orders Over Time</CardTitle></CardHeader>
            <CardContent><div className="h-64 bg-gray-200 rounded-md flex items-center justify-center text-gray-400">Chart Placeholder</div></CardContent>
        </Card>
        <Card>
            <CardHeader><CardTitle>Human Support Requests</CardTitle></CardHeader>
            <CardContent>
                <p className="text-sm text-gray-700 mb-4">Conversations that triggered the 'human_escalation' intent.</p>
                <div className="space-y-2 text-sm">
                    <div className="flex justify-between p-2 rounded hover:bg-gray-50 text-gray-900"><span>Priya Sharma (+91...210)</span><a href="#" className="text-[#ff4d6d] font-medium">View</a></div>
                    <div className="flex justify-between p-2 rounded hover:bg-gray-50 text-gray-900"><span>Rohan Mehra (+91...556)</span><a href="#" className="text-[#ff4d6d] font-medium">View</a></div>
                </div>
            </CardContent>
        </Card>
      </div>
    </div>
  );
};

const HealthPage = () => {
    const [health, setHealth] = React.useState(null);
    const [loading, setLoading] = React.useState(true);

    React.useEffect(() => {
        setTimeout(() => {
            setHealth({
                status: 'healthy',
                services: {
                    database: 'connected',
                    cache: 'connected',
                    whatsapp: 'configured',
                    shopify: 'error',
                },
            });
            setLoading(false);
        }, 1000);
    }, []);

    const StatusIndicator = ({ status }) => {
        const isOk = status === 'connected' || status === 'configured';
        return (
            <div className={`flex items-center gap-2 text-sm font-medium ${isOk ? 'text-green-600' : 'text-red-600'}`}>
                <span className={`h-2.5 w-2.5 rounded-full ${isOk ? 'bg-green-500' : 'bg-red-500'}`}></span>
                {status.replace('_', ' ')}
            </div>
        );
    };

    if (loading) return <div className="text-center p-10 text-gray-900">Loading system health...</div>;

    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold text-black">System Health</h1>
            <Card>
                <CardHeader>
                    <CardTitle>Live Service Status</CardTitle>
                    <CardDescription>Real-time status of all critical services.</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                        {health && Object.entries(health.services).map(([service, status]) => (
                            <div key={service} className="p-4 bg-gray-50 rounded-lg border">
                                <h4 className="font-semibold capitalize text-gray-700">{service}</h4>
                                <StatusIndicator status={status} />
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                    <CardHeader><CardTitle>Circuit Breakers</CardTitle></CardHeader>
                    <CardContent><div className="h-48 bg-gray-200 rounded-md flex items-center justify-center text-gray-400">Circuit Breaker Status Placeholder</div></CardContent>
                </Card>
                <Card>
                    <CardHeader><CardTitle>Recent Critical Alerts</CardTitle></CardHeader>
                    <CardContent><div className="h-48 bg-gray-200 rounded-md flex items-center justify-center text-gray-400">Alerts Feed Placeholder</div></CardContent>
                </Card>
            </div>
        </div>
    );
};

const SecurityPage = () => {
    const [events, setEvents] = React.useState([]);
    const [loading, setLoading] = React.useState(true);

     React.useEffect(() => {
        setTimeout(() => {
            setEvents([
                { _id: '1', event_type: 'successful_login', ip_address: '103.22.11.5', timestamp: new Date().toISOString(), details: { method: 'jwt' } },
                { _id: '2', event_type: 'failed_login', ip_address: '103.22.11.5', timestamp: new Date().toISOString(), details: { reason: 'invalid_password' } },
                { _id: '3', event_type: 'message_broadcast', ip_address: '103.22.11.5', timestamp: new Date().toISOString(), details: { target: 'active', sent_count: 85 } },
            ]);
            setLoading(false);
        }, 1000);
    }, []);

    if (loading) return <div className="text-center p-10 text-gray-900">Loading security events...</div>;

    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold text-black">Security Log</h1>
             <Card>
                <CardHeader>
                    <CardTitle>Recent Security Events</CardTitle>
                    <CardDescription>A log of important security-related activities.</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm text-left text-gray-700">
                            <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                                <tr>
                                    <th scope="col" className="px-6 py-3">Event</th>
                                    <th scope="col" className="px-6 py-3">IP Address</th>
                                    <th scope="col" className="px-6 py-3">Timestamp</th>
                                    <th scope="col" className="px-6 py-3">Details</th>
                                </tr>
                            </thead>
                            <tbody>
                                {events.map(event => (
                                    <tr key={event._id} className="bg-white border-b">
                                        <td className="px-6 py-4 font-medium text-gray-900 capitalize">{event.event_type.replace(/_/g, ' ')}</td>
                                        <td className="px-6 py-4 text-gray-900">{event.ip_address}</td>
                                        <td className="px-6 py-4 text-gray-900">{new Date(event.timestamp).toLocaleString()}</td>
                                        <td className="px-6 py-4 text-xs text-gray-900"><pre>{JSON.stringify(event.details, null, 2)}</pre></td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};

// UPDATED: Broadcast Page with Image Upload
const BroadcastPage = () => {
    const [message, setMessage] = React.useState('');
    const [imageFile, setImageFile] = React.useState(null);
    const [imagePreview, setImagePreview] = React.useState('');
    const [target, setTarget] = React.useState('active');
    const [customGroups, setCustomGroups] = React.useState([{id: 'group1', name: 'Resellers'}, {id: 'group2', name: 'VIP Customers'}]);
    const [selectedGroup, setSelectedGroup] = React.useState('');
    const [isSending, setIsSending] = React.useState(false);
    const [result, setResult] = React.useState(null);

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImageFile(file);
            setImagePreview(URL.createObjectURL(file));
        }
    };

    const clearImage = () => {
        setImageFile(null);
        setImagePreview('');
    };

    const handleSend = () => {
        if ((!message && !imageFile) || isSending) return;
        setIsSending(true);
        setResult(null);
        setTimeout(() => {
            setIsSending(false);
            setResult({ success: true, message: `Broadcast sent with message and ${imageFile ? `image '${imageFile.name}'` : 'no image'}.` });
            setMessage('');
            clearImage();
        }, 2000);
    };

    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold text-black">Broadcast Center</h1>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-2">
                    <Card>
                        <CardHeader>
                            <CardTitle>Compose Broadcast</CardTitle>
                            <CardDescription>Send a bulk message with optional media to a targeted group.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div>
                                <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-1">Message</label>
                                <Textarea id="message" rows="5" placeholder="Type your message here..." value={message} onChange={(e) => setMessage(e.target.value)} />
                                <p className="text-xs text-gray-700 mt-1">{message.length} / 1000 characters</p>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Attach Image (Optional)</label>
                                {imagePreview ? (
                                    <div className="mt-2 relative">
                                        <img src={imagePreview} alt="Preview" className="w-48 h-48 object-cover rounded-md border" />
                                        <button onClick={clearImage} className="absolute top-1 right-1 bg-black/50 text-white rounded-full p-1 leading-none">&times;</button>
                                    </div>
                                ) : (
                                    <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
                                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                            <PaperclipIcon className="w-8 h-8 mb-2 text-gray-500"/>
                                            <p className="mb-1 text-sm text-gray-500"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                                            <p className="text-xs text-gray-500">PNG, JPG, GIF up to 5MB</p>
                                        </div>
                                        <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" />
                                    </label>
                                )}
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">Target Audience</label>
                                <div className="space-y-2">
                                    {['active', 'recent', 'all'].map(t => (
                                        <label key={t} className="flex items-center"><input type="radio" name="target" value={t} checked={target === t} onChange={() => { setTarget(t); setSelectedGroup(''); }} className="h-4 w-4 text-[#ff4d6d] focus:ring-[#ff4d6d] border-gray-300"/><span className="ml-2 text-sm text-gray-900 capitalize">{t} Customers</span></label>
                                    ))}
                                    <label className="flex items-center"><input type="radio" name="target" value="custom" checked={target === 'custom'} onChange={() => setTarget('custom')} className="h-4 w-4 text-[#ff4d6d] focus:ring-[#ff4d6d] border-gray-300"/><span className="ml-2 text-sm text-gray-900">Custom Group</span></label>
                                    {target === 'custom' && (
                                        <select value={selectedGroup} onChange={(e) => setSelectedGroup(e.target.value)} className="w-full mt-2 p-2 border border-gray-300 rounded-md text-gray-900">
                                            <option value="">Select a group</option>
                                            {customGroups.map(g => <option key={g.id} value={g.id}>{g.name}</option>)}
                                        </select>
                                    )}
                                </div>
                            </div>
                            <Button onClick={handleSend} disabled={isSending || (!message && !imageFile)} className="w-full">
                                {isSending ? 'Sending...' : 'Send Broadcast'}
                            </Button>
                            {result && <div className={`p-3 rounded-md text-sm ${result.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>{result.message}</div>}
                        </CardContent>
                    </Card>
                </div>
                <div className="md:col-span-1">
                     <Card>
                        <CardHeader><CardTitle>Manage Custom Groups</CardTitle></CardHeader>
                        <CardContent className="space-y-4">
                            <Input placeholder="New group name..." />
                            <Textarea rows="4" placeholder="Paste phone numbers, one per line..."></Textarea>
                            <Button variant="secondary" className="w-full">Create Group</Button>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
};

// UPDATED: Performance Page with Date Filter, Order ID Tooltip, and Search Functionality
const PerformancePage = () => {
    const [period, setPeriod] = React.useState('daily');
    const [selectedDate, setSelectedDate] = React.useState(new Date().toISOString().split('T')[0]);
    const [performanceData, setPerformanceData] = React.useState([]);
    const [loading, setLoading] = React.useState(true);
    const [searchQuery, setSearchQuery] = React.useState('');
    const [searchResult, setSearchResult] = React.useState(null);

    const allMockData = React.useMemo(() => ({
        daily: [
            { name: 'Swathi', packed: 25, orders: ['#1001', '#1003', '#1005', '#1006', '#1007', '#1009', '#1010', '#1011', '#1012', '#1013'] }, 
            { name: 'Dharam', packed: 22, orders: ['#1002', '#1008', '#1014', '#1015'] }, 
            { name: 'Pushpa', packed: 19, orders: ['#1004', '#1016', '#1017', '#1018', '#1019'] }
        ],
        weekly: [
            { name: 'Swathi', packed: 112, orders: ['#2001', '#2002', '#2003', '#2004', '#2005'] },
            { name: 'Dharam', packed: 105, orders: ['#2006', '#2007', '#2008'] },
            { name: 'Pushpa', packed: 98, orders: ['#2009', '#2010', '#2011', '#2012'] }
        ],
        monthly: [
            { name: 'Swathi', packed: 450, orders: ['#3001', '#3002', '#3003', '#3004', '#3005', '#3006'] },
            { name: 'Dharam', packed: 430, orders: ['#3007', '#3008', '#3009'] },
            { name: 'Pushpa', packed: 410, orders: ['#3010', '#3011', '#3012', '#3013', '#3014'] }
        ],
    }), []);

    React.useEffect(() => {
        setLoading(true);
        setTimeout(() => {
            let data = allMockData[period];
            // Simulate date filtering for daily view
            if (period === 'daily') {
                const today = new Date().toISOString().split('T')[0];
                if (selectedDate === today) {
                    data = allMockData.daily;
                } else {
                    // Show no data for other dates to simulate filtering
                    data = [];
                }
            }
            setPerformanceData(data);
            setLoading(false);
        }, 500);
    }, [period, selectedDate, allMockData]);

    React.useEffect(() => {
        if (searchQuery.trim() === '') {
            setSearchResult(null);
            return;
        }
        for (const packer of performanceData) {
            if (packer.orders && packer.orders.includes(searchQuery.trim())) {
                setSearchResult(packer.name);
                return;
            }
        }
        setSearchResult('not found');
    }, [searchQuery, performanceData]);

    const sortedPerformanceData = React.useMemo(() => 
        [...performanceData].sort((a, b) => b.packed - a.packed),
        [performanceData]
    );

    // Component to render order IDs with a tooltip for long lists
    const OrderList = ({ orders }) => {
        if (!orders || orders.length === 0) return 'N/A';

        const displayLimit = 2;
        const truncatedOrders = orders.slice(0, displayLimit).join(', ');
        const remainingCount = orders.length - displayLimit;

        if (orders.length <= displayLimit) {
            return orders.join(', ');
        }
        
        return (
            <div className="relative group">
                <span>{truncatedOrders}, </span>
                <span className="font-bold cursor-pointer text-[#ff4d6d]">
                    +{remainingCount} more
                </span>
                <div className="absolute hidden group-hover:block bottom-full mb-2 w-max max-w-xs bg-gray-800 text-white text-xs rounded-md p-2 z-10 break-words">
                    {orders.join(', ')}
                </div>
            </div>
        );
    };

    return (
        <div className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
                <h1 className="text-3xl font-bold text-black">Packer Performance</h1>
                <div className="flex items-center gap-2 rounded-lg bg-white p-1 border">
                    {['daily', 'weekly', 'monthly'].map(p => (
                        <Button key={p} onClick={() => setPeriod(p)} variant={period === p ? 'primary' : 'ghost'} className="capitalize px-3 py-1 text-sm">
                            {p}
                        </Button>
                    ))}
                    {period === 'daily' && (
                        <Input type="date" className="h-8 text-sm" value={selectedDate} onChange={e => setSelectedDate(e.target.value)}/>
                    )}
                </div>
            </div>
            <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">Search by Order ID</label>
                <Input 
                    type="text" 
                    placeholder="Enter order ID, e.g., #1001" 
                    value={searchQuery} 
                    onChange={(e) => setSearchQuery(e.target.value)} 
                    className="w-full max-w-md"
                />
                {searchResult && (
                    <p className={`mt-2 text-sm font-medium ${searchResult === 'not found' ? 'text-red-600' : 'text-green-600'}`}>
                        {searchResult === 'not found' ? 'Order not found in the current data.' : `Packed by: ${searchResult}`}
                    </p>
                )}
            </div>
            <Card>
                <CardHeader>
                    <CardTitle>Orders Packed ({period})</CardTitle>
                    <CardDescription>Leaderboard of orders packed by each team member.</CardDescription>
                </CardHeader>
                <CardContent>
                    {loading ? <div className="text-center p-8 text-gray-900">Loading performance data...</div> : (
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm text-left text-gray-700">
                                <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                                    <tr>
                                        <th scope="col" className="px-6 py-3">Rank</th>
                                        <th scope="col" className="px-6 py-3">Packer Name</th>
                                        <th scope="col" className="px-6 py-3">Parcels Packed</th>
                                        <th scope="col" className="px-6 py-3">Order IDs</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {sortedPerformanceData.length > 0 ? sortedPerformanceData.map((packer, index) => (
                                        <tr key={packer.name} className="bg-white border-b">
                                            <td className="px-6 py-4 font-medium text-gray-900">{index + 1}</td>
                                            <td className="px-6 py-4 text-gray-900">{packer.name}</td>
                                            <td className="px-6 py-4 font-bold text-[#ff4d6d]">{packer.packed}</td>
                                            <td className="px-6 py-4 text-xs text-gray-700">
                                                <OrderList orders={packer.orders} />
                                            </td>
                                        </tr>
                                    )) : (
                                        <tr>
                                            <td colSpan="4" className="text-center p-8 text-gray-500">No performance data for the selected period/date.</td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
};

const ConversationsPage = () => {
    const [conversations, setConversations] = React.useState([]);
    const [selectedConvo, setSelectedConvo] = React.useState(null);
    const [messages, setMessages] = React.useState([]);
    const [loading, setLoading] = React.useState(true);

    React.useEffect(() => {
        setTimeout(() => {
            setConversations([
                { id: 1, name: 'Priya Sharma', phone: '+919876543210', lastMessage: 'Is this available in gold?', time: '5m ago', status: 'delivered' },
                { id: 2, name: 'Amit Singh', phone: '+919123456789', lastMessage: 'Thank you!', time: '2h ago', status: 'read' },
                { id: 3, name: 'Sneha Reddy', phone: '+918765432109', lastMessage: 'What is the price for #1234?', time: '1d ago', status: 'sent' },
            ]);
            setLoading(false);
        }, 1000);
    }, []);
    
    const handleSelectConvo = (convo) => {
        setSelectedConvo(convo);
        setMessages([
            { from: 'user', text: 'Hi, I saw a necklace I liked.' },
            { from: 'bot', text: 'Hello! I can help with that. Could you describe it or send a picture?' },
            { from: 'user', text: 'It was a gold choker with red stones.' },
            { from: 'bot', text: 'Searching for gold chokers with red stones for you now... ✨' },
        ]);
    };

    const StatusTick = ({ status }) => {
        if (status === 'read') return <span className="text-blue-500">✓✓</span>;
        if (status === 'delivered') return <span className="text-gray-900">✓✓</span>;
        return <span className="text-gray-900">✓</span>;
    };

    return (
        <div className="h-[calc(100vh-84px)] flex flex-col">
            <h1 className="text-3xl font-bold text-black pb-4">Conversation Explorer</h1>
            <div className="flex-grow grid grid-cols-1 lg:grid-cols-3 gap-6 overflow-hidden">
                <Card className="lg:col-span-1 flex flex-col">
                    <CardHeader><CardTitle>Recent Chats</CardTitle></CardHeader>
                    <CardContent className="flex-grow overflow-y-auto p-2">
                        {loading ? <p className="text-gray-900">Loading...</p> : (
                            <div className="space-y-2">
                                {conversations.map(convo => (
                                    <div key={convo.id} onClick={() => handleSelectConvo(convo)}
                                        className={`p-3 rounded-lg cursor-pointer transition-colors ${selectedConvo?.id === convo.id ? 'bg-[#ff4d6d] text-white' : 'hover:bg-gray-100'}`}>
                                        <div className="flex justify-between items-center">
                                            <p className={`font-semibold ${selectedConvo?.id === convo.id ? 'text-white' : 'text-gray-900'}`}>{convo.name}</p>
                                            <p className={`text-xs ${selectedConvo?.id === convo.id ? 'text-gray-200' : 'text-gray-700'}`}>{convo.time}</p>
                                        </div>
                                        <div className="flex justify-between items-center">
                                            <p className={`text-sm truncate ${selectedConvo?.id === convo.id ? 'text-gray-200' : 'text-gray-700'}`}>{convo.lastMessage}</p>
                                            <StatusTick status={convo.status} />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </CardContent>
                </Card>
                <Card className="lg:col-span-2 flex flex-col">
                    <CardHeader>
                        <CardTitle>{selectedConvo ? selectedConvo.name : 'Select a Conversation'}</CardTitle>
                        <CardDescription>{selectedConvo ? selectedConvo.phone : 'No chat selected'}</CardDescription>
                    </CardHeader>
                    <CardContent className="flex-grow overflow-y-auto bg-gray-50">
                        {selectedConvo ? (
                            <div className="space-y-4 text-gray-900">
                                {messages.map((msg, i) => (
                                    <div key={i} className={`flex ${msg.from === 'user' ? 'justify-end' : 'justify-start'}`}>
                                        <div className={`max-w-xs lg:max-w-md p-3 rounded-lg ${msg.from === 'user' ? 'bg-[#ff4d6d] text-white' : 'bg-white border'}`}>
                                            {msg.text}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="flex items-center justify-center h-full text-gray-700">
                                <p>Select a conversation from the left to view the transcript.</p>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};


// --- RULES ENGINE PAGE ---
const RulesEnginePage = () => {
  const [intents, setIntents] = React.useState([
    { id: 1, name: "latest_arrivals_inquiry", keywords: ["latest", "new", "newest", "recent"], phrases: ["just added"] },
    { id: 2, name: "bestseller_inquiry", keywords: ["bestseller", "popular", "trending"], phrases: ["best selling", "top selling"] },
    { id: 3, name: "human_escalation", keywords: ["human", "agent", "person"], phrases: ["talk to human", "speak to a person"] },
  ]);

  const [keywords, setKeywords] = React.useState(["ruby", "necklace", "earring", "bangle", "bracelet", "ring", "pendant", "gold", "silver", "diamond"]);
  const [testInput, setTestInput] = React.useState("");
  const [testResult, setTestResult] = React.useState(null);

  function handleAddIntent() {
    setIntents([...intents, { id: Date.now(), name: "new_intent", keywords: [], phrases: [] }]);
  }
  
  const handleIntentChange = (index, field, value) => {
    setIntents(currentIntents => 
        currentIntents.map((intent, i) => {
            if (i === index) {
                return { ...intent, [field]: value };
            }
            return intent;
        })
    );
  };

  function handleTest() {
    if (!testInput.trim()) {
        setTestResult({ message: "Please enter a test query.", type: 'error' });
        return;
    }
    const matched = intents.find(i => 
        i.keywords.some(k => testInput.toLowerCase().includes(k)) ||
        i.phrases.some(p => testInput.toLowerCase().includes(p))
    );
    setTestResult(matched 
        ? { message: `Matched Intent: ${matched.name}`, type: 'success' }
        : { message: "No intent matched.", type: 'error' }
    );
  }

  return (
    <div className="space-y-6">
        <div className="flex items-center justify-between">
            <div>
                <h1 className="text-3xl font-bold text-black">Rules Engine</h1>
                <p className="text-gray-700 mt-1">Manage AI intents, keywords, and test matching logic.</p>
            </div>
            <Button variant="success">Deploy Changes</Button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
                <Card>
                    <CardHeader>
                        <CardTitle>Intent Rules</CardTitle>
                        <CardDescription>Define rules to understand user messages. Rules are processed by priority.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        {intents.map((intent, idx) => (
                            <div key={intent.id} className="p-4 border rounded-lg bg-gray-50/50">
                                <Input
                                    className="text-md font-semibold mb-3 bg-white"
                                    value={intent.name}
                                    onChange={e => handleIntentChange(idx, 'name', e.target.value)}
                                />
                                <div className="space-y-2">
                                    <label className="text-sm font-medium text-gray-700">Single Keywords</label>
                                    <TagInput 
                                        tags={intent.keywords} 
                                        setTags={(newTags) => handleIntentChange(idx, 'keywords', newTags)}
                                        placeholder="Add keyword..." 
                                    />
                                </div>
                                <div className="mt-4 space-y-2">
                                    <label className="text-sm font-medium text-gray-700">Multi-Word Phrases</label>
                                    <TagInput 
                                        tags={intent.phrases} 
                                        setTags={(newTags) => handleIntentChange(idx, 'phrases', newTags)}
                                        placeholder="Add phrase..." 
                                    />
                                </div>
                            </div>
                        ))}
                        <Button onClick={handleAddIntent} variant="secondary" className="w-full">
                            Add New Intent
                        </Button>
                    </CardContent>
                </Card>
            </div>

            <div className="lg:col-span-1 space-y-6">
                 <Card>
                    <CardHeader>
                        <CardTitle>Test Rule Matching</CardTitle>
                        <CardDescription>Simulate a user message to see which intent it matches.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <Input
                            placeholder="e.g., 'show me new necklaces'"
                            value={testInput}
                            onChange={e => setTestInput(e.target.value)}
                        />
                        <Button onClick={handleTest} className="w-full">Test</Button>
                        {testResult && (
                            <div className={`p-3 rounded-md text-sm ${testResult.type === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                                {testResult.message}
                            </div>
                        )}
                    </CardContent>
                </Card>
                 <Card>
                    <CardHeader>
                        <CardTitle>Keyword Dictionary</CardTitle>
                        <CardDescription>Vocabulary of known product terms for matching.</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <TagInput tags={keywords} setTags={setKeywords} placeholder="Add keyword..."/>
                    </CardContent>
                </Card>
            </div>
        </div>
    </div>
  );
};

// --- STRINGS MANAGER PAGE ---
const StringsManagerPage = () => {
  const [contactInfo, setContactInfo] = React.useState(`👋 Here is our official contact information:

📍 **Business Address:**
FeelOri
Sai Nidhi, Plot 9, Krishnapuri Colony,
Secunderabad, Hyderabad, Telangana 500026, India

📧 **Email Support:**
support@feelori.com`);

  const [resellerInfo, setResellerInfo] = React.useState(`We're excited you're interested in partnering with us! ✨

We offer a fantastic **Reseller Program** and an exclusive **WhatsApp Broadcast Group**.

To get more details, please contact our team directly:
📞 **WhatsApp/Call:** 7337294499`);

  const [shippingInfo, setShippingInfo] = React.useState(`🚚 **FeelOri Shipping Policy**

📍 **Metro & Tier-1 Cities:** 3-5 business days
📍 **Remote & Rural Areas:** 5-7 business days

📦 **Shipping Costs**
🎉 **Free Shipping** on all orders above ₹999!`);


  function handleSave() {
    alert("Strings saved! (This would typically call a backend API).");
  }
  
  function handleRollback() {
    alert("Changes reverted! (This would fetch the last saved version).");
  }

  return (
    <div className="space-y-6">
        <div className="flex items-center justify-between">
            <div>
                <h1 className="text-3xl font-bold text-black">Strings Manager</h1>
                <p className="text-gray-700 mt-1">Edit all user-facing text and messages from one place.</p>
            </div>
            <div className="flex gap-2">
                <Button onClick={handleSave} variant="primary">Save Changes</Button>
                <Button onClick={handleRollback} variant="secondary">Rollback</Button>
            </div>
        </div>

        <Card>
            <CardHeader>
                <CardTitle>Contact & Business Info</CardTitle>
                <CardDescription>Update official contact details and business information.</CardDescription>
            </CardHeader>
            <CardContent className="grid md:grid-cols-2 gap-6">
                 <div>
                    <label className="text-sm font-medium text-gray-700 mb-2 block">CONTACT_INFO</label>
                    <Textarea rows={10} value={contactInfo} onChange={e => setContactInfo(e.target.value)} />
                </div>
                <div>
                    <label className="text-sm font-medium text-gray-700 mb-2 block">RESELLER_INFO</label>
                    <Textarea rows={10} value={resellerInfo} onChange={e => setResellerInfo(e.target.value)} />
                </div>
            </CardContent>
        </Card>

        <Card>
            <CardHeader>
                <CardTitle>Policies & Support Responses</CardTitle>
                <CardDescription>Manage shipping details and standard support replies.</CardDescription>
            </CardHeader>
             <CardContent>
                <label className="text-sm font-medium text-gray-700 mb-2 block">SHIPPING_POLICY_INFO</label>
                <Textarea rows={8} value={shippingInfo} onChange={e => setShippingInfo(e.target.value)} />
            </CardContent>
        </Card>
    </div>
  );
};


// --- 6. LOGIN PAGE (app/auth/login/page.tsx) ---

const LoginPage = ({ onLogin }) => {
  const [password, setPassword] = React.useState('');
  const [error, setError] = React.useState('');
  const [loading, setLoading] = React.useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (loading) return;
    setLoading(true);
    setError('');

    setTimeout(() => {
      if (password === 'admin123') {
        onLogin('fake_jwt_token');
      } else {
        setError('Invalid credentials. Please try again.');
        setLoading(false);
      }
    }, 1000);
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50">
      <div className="w-full max-w-md">
        <Card>
          <CardHeader className="text-center">
            <h1 className="text-2xl font-bold text-black">FeelOri Admin Login</h1>
            <p className="text-gray-700">Enter your password to access the dashboard.</p>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <Input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              {error && <p className="text-sm text-red-600">{error}</p>}
              <Button type="submit" disabled={loading} className="w-full">
                {loading ? 'Logging in...' : 'Login'}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

// --- 7. MAIN APP COMPONENT (app/layout.tsx & page.tsx) ---

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = React.useState(false);
  const [activePage, setActivePage] = React.useState('dashboard');

  React.useEffect(() => {
    const token = localStorage.getItem('feelori_admin_token');
    if (token) {
      setIsAuthenticated(true);
    }
  }, []);

  const handleLogin = (token) => {
    localStorage.setItem('feelori_admin_token', token);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem('feelori_admin_token');
    setIsAuthenticated(false);
  };

  if (!isAuthenticated) {
    return <LoginPage onLogin={handleLogin} />;
  }

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard':
        return <DashboardPage setPage={setActivePage} />;
      case 'conversations':
        return <ConversationsPage />;
      case 'performance':
        return <PerformancePage />;
      case 'health':
        return <HealthPage />;
      case 'security':
        return <SecurityPage />;
      case 'broadcast':
        return <BroadcastPage />;
      case 'rules':
        return <RulesEnginePage />;
      case 'strings':
        return <StringsManagerPage />;
      default:
        return <DashboardPage setPage={setActivePage} />;
    }
  };

  return (
    <DashboardLayout activePage={activePage} setActivePage={setActivePage} onLogout={handleLogout}>
      {renderPage()}
    </DashboardLayout>
  );
}