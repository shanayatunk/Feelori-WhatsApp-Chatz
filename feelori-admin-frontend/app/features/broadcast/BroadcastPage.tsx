import React, { useState, useEffect } from 'react';
import { apiService, BroadcastGroup } from '../../../lib/api'; // Import BroadcastGroup
import { Button } from '../../components/ui/Button';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';
import { Input } from '../../components/ui/Input';
import { Textarea } from '../../components/ui/Textarea';
import { BroadcastHistoryPage } from './BroadcastHistoryPage';
import { BroadcastReportPage } from './BroadcastReportPage';

// --- NEW Group Management Component ---
const ManageGroups = ({ groups, onGroupCreated }: { groups: BroadcastGroup[], onGroupCreated: () => void }) => {
    const [newGroupName, setNewGroupName] = useState('');
    const [newGroupPhones, setNewGroupPhones] = useState('');
    const [isCreating, setIsCreating] = useState(false);
    const [createError, setCreateError] = useState<string | null>(null);
    const [createSuccess, setCreateSuccess] = useState<string | null>(null);

    const handleCreateGroup = async () => {
        if (!newGroupName.trim() || !newGroupPhones.trim() || isCreating) return;
        setIsCreating(true);
        setCreateError(null);
        setCreateSuccess(null);
        // Split by newline or comma, trim whitespace, remove empty entries
        const phoneList = newGroupPhones
            .split(/[\n,]+/)
            .map(p => p.trim())
            .filter(Boolean);

        if (phoneList.length === 0) {
            setCreateError("Please enter at least one valid phone number.");
            setIsCreating(false);
            return;
        }

        try {
            await apiService.createBroadcastGroup(newGroupName.trim(), phoneList);
            setCreateSuccess(`Group "${newGroupName.trim()}" created successfully!`);
            setNewGroupName('');
            setNewGroupPhones('');
            onGroupCreated(); // Notify parent to refresh group list
            // Clear success message after a delay
            setTimeout(() => setCreateSuccess(null), 3000);
        } catch (err) {
            setCreateError(err instanceof Error ? err.message : 'Failed to create group.');
             // Clear error message after a delay
             setTimeout(() => setCreateError(null), 5000);
        } finally {
            setIsCreating(false);
        }
    };

    return (
        <Card>
            <CardHeader><CardTitle>Manage Custom Groups</CardTitle></CardHeader>
            <CardContent className="space-y-4">
                <Input
                    placeholder="New group name..."
                    value={newGroupName}
                    onChange={(e) => setNewGroupName(e.target.value)}
                />
                <Textarea
                    rows={4}
                    placeholder="Paste phone numbers, one per line or comma-separated..."
                    value={newGroupPhones}
                    onChange={(e) => setNewGroupPhones(e.target.value)}
                />
                <Button
                    variant="secondary"
                    className="w-full"
                    onClick={handleCreateGroup}
                    disabled={isCreating || !newGroupName.trim() || !newGroupPhones.trim()}
                >
                    {isCreating ? 'Creating...' : 'Create Group'}
                </Button>
                {createError && <p className="text-sm text-red-600 p-2 bg-red-50 rounded">{createError}</p>}
                {createSuccess && <p className="text-sm text-green-600 p-2 bg-green-50 rounded">{createSuccess}</p>}

                {/* Display existing groups */}
                {groups.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-gray-200">
                        <h4 className="text-sm font-medium text-gray-900 mb-2">Existing Groups</h4>
                        <ul className="space-y-1 text-sm text-gray-700 max-h-40 overflow-y-auto pr-2">
                            {groups.map(group => (
                                <li key={group.id} className="flex justify-between items-center p-1 hover:bg-gray-50 rounded">
                                    <span className="truncate mr-2" title={group.name}>{group.name}</span>
                                    <span className="flex-shrink-0 text-xs bg-gray-200 text-gray-700 px-1.5 py-0.5 rounded">
                                        {group.phone_count}
                                    </span>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </CardContent>
        </Card>
    );
};

// --- MODIFIED ComposeView Component ---
const ComposeView = ({ groups, onGroupCreated }: { groups: BroadcastGroup[], onGroupCreated: () => void }) => {
    const [message, setMessage] = useState('');
    const [imageUrl, setImageUrl] = useState('');
    // Add 'custom_group' to possible targets
    const [target, setTarget] = useState<'all' | 'active' | 'inactive' | 'custom_group'>('active');
    // State for selected custom group
    const [selectedGroupId, setSelectedGroupId] = useState<string>('');
    const [isSending, setIsSending] = useState(false);
    const [result, setResult] = useState<{success: boolean, message: string} | null>(null);

    // Effect to set default group ID when custom_group becomes available or is selected
     useEffect(() => {
        if (target === 'custom_group') {
            // If custom group is selected and no group ID is set yet,
            // or if the currently selected ID is no longer valid, select the first available group.
            if ((!selectedGroupId || !groups.find(g => g.id === selectedGroupId)) && groups.length > 0) {
                 setSelectedGroupId(groups[0].id);
            }
        } else {
             // If target is not custom_group, clear the selected group ID
             setSelectedGroupId('');
        }
    // Dependency includes groups to handle cases where groups load after initial render
    }, [target, groups, selectedGroupId]);


    const handleSend = async () => {
        if (!message.trim()) {
             setResult({ success: false, message: "Message cannot be empty." });
             return;
        }
        if (isSending) return;

        if (target === 'custom_group' && !selectedGroupId) {
            setResult({ success: false, message: "Please select a custom group." });
            return;
        }

        setIsSending(true);
        setResult(null);

        try {
            // Pass targetGroupId if 'custom_group' is selected
            const response = await apiService.broadcast(
                message.trim(), // Send trimmed message
                target,
                imageUrl.trim() || undefined, // Send undefined if empty or only whitespace
                undefined, // targetPhones - not used in this UI yet
                target === 'custom_group' ? selectedGroupId : undefined
            );
            // Include Job ID in success message
            setResult({ success: true, message: `${response.message} (Job ID: ${response.job_id || 'N/A'})` });
            setMessage('');
            setImageUrl('');
            // Optional: reset target or keep it
            // setTarget('active');
            // setSelectedGroupId('');
            setTimeout(() => setResult(null), 5000); // Clear success message after 5s

        } catch (err) {
            setResult({ success: false, message: err instanceof Error ? err.message : 'An unknown error occurred sending the broadcast.'});
             // Clear error message after a delay
             setTimeout(() => setResult(null), 7000);
        } finally {
            setIsSending(false);
        }
    };


    const targetOptions = ['all', 'active', 'inactive']; // Add 'custom_group' below conditionally

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-2">
                <Card>
                    <CardHeader>
                        <CardTitle>Compose Message</CardTitle>
                        <CardDescription>Send a message to a group of customers. Use placeholders like {'{{name}}'}.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <Textarea
                            rows={6}
                            placeholder="Your message here..."
                            value={message}
                            onChange={(e) => setMessage(e.target.value)}
                        />
                        <Input
                            type="url" // Use URL type for better validation potentially
                            placeholder="Optional: Image URL (e.g., https://.../image.png)"
                            value={imageUrl}
                            onChange={(e) => setImageUrl(e.target.value)}
                        />
                         <div>
                            <label className="text-sm font-medium text-gray-900 mb-2 block">Target Audience</label>
                            <div className="space-y-2">
                                {targetOptions.map((t) => (
                                    <label key={t} className="flex items-center cursor-pointer p-1">
                                        <input
                                            type="radio"
                                            value={t}
                                            checked={target === t}
                                            onChange={() => setTarget(t as any)} // Cast needed here
                                            className="h-4 w-4 text-[#ff4d6d] focus:ring-[#ff4d6d] border-gray-300"
                                        />
                                        <span className="ml-2 text-sm text-gray-900 capitalize">{t} Customers</span>
                                    </label>
                                ))}
                                {/* Add Custom Group Radio and Select */}
                                {groups.length > 0 && (
                                    <label key="custom_group" className="flex items-center cursor-pointer p-1">
                                        <input
                                            type="radio"
                                            value="custom_group"
                                            checked={target === 'custom_group'}
                                            onChange={() => setTarget('custom_group')}
                                            className="h-4 w-4 text-[#ff4d6d] focus:ring-[#ff4d6d] border-gray-300"
                                        />
                                        <span className="ml-2 text-sm text-gray-900">Custom Group:</span>
                                        <select
                                            value={selectedGroupId}
                                            onChange={(e) => setSelectedGroupId(e.target.value)}
                                            // Only truly disable if target is not custom group,
                                            // otherwise allow selection even if default is already set
                                            disabled={target !== 'custom_group'}
                                            className={`ml-2 flex-grow rounded-md border border-gray-300 bg-white px-2 py-1 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-[#ff4d6d] focus:ring-offset-1 ${target !== 'custom_group' ? 'opacity-50 cursor-not-allowed bg-gray-100' : ''}`}
                                        >
                                            {/* Default empty option only if no group is selected */}
                                            {!selectedGroupId && <option value="" disabled>Select Group</option>}
                                            {groups.map(group => (
                                                <option key={group.id} value={group.id}>
                                                    {group.name} ({group.phone_count})
                                                </option>
                                            ))}
                                        </select>
                                    </label>
                                )}
                                {groups.length === 0 && (
                                     <p className="text-xs text-gray-500 pl-6">No custom groups created yet.</p>
                                )}
                            </div>
                        </div>
                        <Button
                            onClick={handleSend}
                            disabled={isSending || !message.trim()} // Disable if message is empty/whitespace
                            className="w-full"
                         >
                            {isSending ? 'Sending...' : 'Send Broadcast'}
                        </Button>
                        {/* Improved result display */}
                        {result && (
                          <div className={`p-3 rounded-md text-sm font-medium ${result.success ? 'bg-green-100 text-green-800 border border-green-200' : 'bg-red-100 text-red-800 border border-red-200'}`}>
                              {result.message}
                           </div>
                         )}
                    </CardContent>
                </Card>
            </div>
            {/* Use the new ManageGroups component */}
            <div className="md:col-span-1">
                 <ManageGroups groups={groups} onGroupCreated={onGroupCreated} />
            </div>
        </div>
    );
};


// --- MODIFIED BroadcastPage Parent Component ---
export const BroadcastPage = () => {
    const [view, setView] = useState<'compose' | 'history' | 'report'>('compose');
    const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
    // State to hold the list of custom groups
    const [groups, setGroups] = useState<BroadcastGroup[]>([]);
    const [loadingGroups, setLoadingGroups] = useState(true); // Loading state for groups
    const [groupsError, setGroupsError] = useState<string | null>(null); // Error state for groups

    // Function to fetch groups
    const fetchGroups = async () => {
        // Don't reset groups immediately, only on success
        // setGroups([]);
        setGroupsError(null);
        setLoadingGroups(true);
        try {
            const fetchedGroups = await apiService.getBroadcastGroups();
            setGroups(fetchedGroups);
        } catch (error) {
            console.error("Failed to fetch broadcast groups:", error);
            setGroupsError(error instanceof Error ? error.message : "Failed to load groups.");
        } finally {
             setLoadingGroups(false);
        }
    };

    // Fetch groups on initial mount and when view changes back to compose
    useEffect(() => {
        // Only fetch if in compose view
        if (view === 'compose') {
            fetchGroups();
        }
        // No dependency on fetchGroups itself needed if defined within scope or stable
    }, [view]);

    const handleSelectBroadcast = (jobId: string) => {
        setSelectedJobId(jobId);
        setView('report');
    };

    // Function to handle navigating back to compose, ensuring groups are fetched
    const backToCompose = () => {
        setSelectedJobId(null);
        setView('compose');
        // Fetch groups again when returning to compose view
        // fetchGroups(); // Already handled by useEffect on [view]
    };

    // Function to handle navigating back from report to history
     const backToHistory = () => {
        setSelectedJobId(null);
        setView('history');
     };


    const renderContent = () => {
        switch (view) {
            case 'history':
                return <BroadcastHistoryPage onSelectBroadcast={handleSelectBroadcast} />;
            case 'report':
                // Pass onBack to navigate back to history view
                return <BroadcastReportPage jobId={selectedJobId!} onBack={backToHistory} />;
            case 'compose':
            default:
                 // Show loading or error specifically for groups if applicable
                 if (loadingGroups) return <div className="text-center p-4">Loading groups...</div>;
                 if (groupsError) return <div className="text-center p-4 text-red-600">Error loading groups: {groupsError}</div>;
                 // Pass groups and the fetchGroups function (for refresh) to ComposeView
                return <ComposeView groups={groups} onGroupCreated={fetchGroups} />;
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center flex-wrap gap-2"> {/* Added flex-wrap */}
                <h1 className="text-3xl font-bold text-black">Broadcast Center</h1>
                {/* Adjust button visibility/text based on current view */}
                {view === 'compose' && (
                     <Button onClick={() => setView('history')} variant="secondary">View History</Button>
                )}
                 {/* Show Back to Compose button when not on compose view */}
                 {view !== 'compose' && (
                     <Button onClick={backToCompose} variant="secondary">← Back to Compose</Button>
                 )}
            </div>
            {/* Render content - loading/error handled within renderContent for compose view */}
            {renderContent()}
        </div>
    );
};