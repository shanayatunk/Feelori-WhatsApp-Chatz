// /feelori-admin-frontend/app/features/triage/TriagePage.tsx

"use client";
import React from 'react';
// 1. Import 'mutate' from SWR to refresh data after an update
import useSWR, { mutate } from 'swr';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
// 2. Import the new 'resolveTriageTicket' function
import { getTriageTickets, TriageTicket, resolveTriageTicket } from '../../../lib/api';
import { Button } from '../../components/ui/Button';

// A helper to format dates
const formatDateTime = (isoString: string) => {
  try {
    return new Date(isoString).toLocaleString('en-IN', {
      day: 'numeric',
      month: 'short',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  } catch (e) {
    return 'Invalid Date';
  }
};

export const TriagePage = () => {
  const { data, error, isLoading } = useSWR('triageTickets', getTriageTickets);

  // 3. Add the handler function to resolve a ticket
  const handleResolveTicket = async (ticketId: string) => {
    try {
      // Call the API to mark the ticket as resolved
      await resolveTriageTicket(ticketId);
      // Tell SWR to re-fetch the data, which will remove the resolved
      // ticket from the list automatically.
      mutate('triageTickets');
    } catch (err) {
      console.error("Failed to resolve ticket:", err);
      // Optional: Add a user-facing error notification here
      alert("Could not resolve the ticket. Please try again.");
    }
  };

  // 4. Add a handler for viewing photos (connect to backend later)
  const handleViewPhoto = (mediaId: string) => {
    // This will open a new tab to the backend endpoint which then redirects
    // to the temporary WhatsApp media URL.
    window.open(`/api/triage/media/${mediaId}`, '_blank');
  };


  const renderContent = () => {
    if (isLoading) {
      return <div className="text-center p-8">Loading tickets...</div>;
    }
    if (error) {
      return <div className="text-center p-8 text-red-600">Failed to load tickets. Please try again later.</div>;
    }
    if (!data || data.tickets.length === 0) {
      return <div className="text-center p-8 text-gray-500">No pending triage tickets. All caught up!</div>;
    }

    return (
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Reported</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Order #</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Customer Phone</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Issue</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data.tickets.map((ticket) => (
              <tr key={ticket._id}>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{formatDateTime(ticket.created_at)}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{ticket.order_number}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{ticket.customer_phone}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{ticket.issue_type}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium space-x-2">
                  {/* --- 5. THE FIX IS APPLIED HERE --- */}
                  {ticket.image_media_id && ticket.image_media_id !== 'N/A' && (
                    <Button
                      variant="secondary"
                      onClick={() => handleViewPhoto(ticket.image_media_id!)}
                    >
                      View Photo
                    </Button>
                  )}
                  <Button
                    variant="primary"
                    onClick={() => handleResolveTicket(ticket._id)}
                  >
                    Resolve
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="p-4 md:p-8 space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Triage Tickets</h1>
      <Card>
        <CardHeader>
          <CardTitle>Pending Customer Issues</CardTitle>
        </CardHeader>
        <CardContent>
          {renderContent()}
        </CardContent>
      </Card>
    </div>
  );
};
