// TriagePage.tsx
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
  const [selectedImage, setSelectedImage] = React.useState<string | null>(null);

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

  const handleViewPhoto = async (mediaId: string) => {
    try {
      const token = localStorage.getItem('feelori_admin_token');
      const photoUrl = getTriageMediaUrl(mediaId);
      
      const response = await fetch(photoUrl, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to fetch image');
      
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setSelectedImage(imageUrl);
    } catch (err) {
      console.error("Failed to load image:", err);
      alert("Could not load the image. Please try again.");
    }
  };

  const closeImageModal = () => {
    if (selectedImage) {
      URL.revokeObjectURL(selectedImage);
      setSelectedImage(null);
    }
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

      {/* Image Modal */}
      {selectedImage && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
          onClick={closeImageModal}
        >
          <div className="relative max-w-4xl max-h-screen p-4">
            <button
              onClick={closeImageModal}
              className="absolute top-2 right-2 bg-white rounded-full w-8 h-8 flex items-center justify-center hover:bg-gray-200 text-gray-700 font-bold"
            >
              ✕
            </button>
            <img 
              src={selectedImage} 
              alt="Ticket photo" 
              className="max-w-full max-h-screen object-contain"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </div>
  );
};