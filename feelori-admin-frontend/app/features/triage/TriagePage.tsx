// TriagePage.tsx with debugging
"use client";
import React from 'react';
import useSWR, { mutate } from 'swr';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { getTriageTickets, TriageTicket, resolveTriageTicket, getTriageMediaUrl } from '../../../lib/api';
import { Button } from '../../components/ui/Button';

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
  // Add debugging
  console.log('Environment check:', {
    NODE_ENV: process.env.NODE_ENV,
    API_URL: process.env.NEXT_PUBLIC_API_URL,
    CURRENT_PROTOCOL: window.location.protocol,
    CURRENT_HOST: window.location.host
  });

  const { data, error, isLoading } = useSWR('triageTickets', getTriageTickets);

  const handleResolveTicket = async (ticketId: string) => {
    try {
      console.log('Resolving ticket:', ticketId);
      await resolveTriageTicket(ticketId);
      console.log('Ticket resolved successfully');
      mutate('triageTickets');
    } catch (err) {
      console.error("Failed to resolve ticket:", err);
      alert("Could not resolve the ticket. Please try again.");
    }
  };

  const handleViewPhoto = (mediaId: string) => {
    const photoUrl = getTriageMediaUrl(mediaId);
    console.log('Opening photo URL:', photoUrl);
    window.open(photoUrl, '_blank');
  };

  const renderContent = () => {
    if (isLoading) {
      return <div className="text-center p-8">Loading tickets...</div>;
    }
    
    if (error) {
      console.error('Triage page error:', error);
      return (
        <div className="text-center p-8 text-red-600">
          <p>Failed to load tickets.</p>
          <p className="text-sm mt-2">Error: {error.message}</p>
          <p className="text-xs mt-1 text-gray-500">Check console for more details</p>
        </div>
      );
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