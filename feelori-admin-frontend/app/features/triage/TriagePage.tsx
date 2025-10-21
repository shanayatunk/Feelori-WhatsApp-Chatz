// TriagePage.tsx
"use client";
import React, { useState, useEffect, useCallback } from 'react'; // Import useCallback
// FIX: Remove unused TriageTicket import if it's defined locally and not used
// import { TriageTicket } from '../../../lib/api';
import useSWR, { mutate } from 'swr';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import { getTriageTickets, resolveTriageTicket, getTriageMediaUrl, apiService } from '../../../lib/api'; // Import apiService
import { Button } from '../../components/ui/Button';
// Note: If TriageTicket type is needed, import it from api.ts
import type { TriageTicket } from '../../../lib/api'; // Use type import

const formatDateTime = (isoString: string) => {
  try {
    return new Date(isoString).toLocaleString('en-IN', {
      day: 'numeric',
      month: 'short',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  } catch (e) { // FIX: Disable eslint warning for unused 'e'
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    return 'Invalid Date';
  }
};

export const TriagePage = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isImageLoading, setIsImageLoading] = useState(false); // Add loading state for image

  // Fetch tickets using SWR
  const { data, error: ticketsError, isLoading: isLoadingTickets } = useSWR< { tickets: TriageTicket[] } >('triageTickets', getTriageTickets);

  const handleResolveTicket = async (ticketId: string) => {
    try {
      console.log('Resolving ticket:', ticketId);
      // Optimistic UI update: Remove ticket locally first
      mutate('triageTickets', (currentData: { tickets: TriageTicket[] } | undefined) => {
          if (!currentData) return currentData;
          return { tickets: currentData.tickets.filter(t => t._id !== ticketId) };
      }, false); // false means don't revalidate yet

      await resolveTriageTicket(ticketId);
      console.log('Ticket resolved successfully');
      // Revalidate to confirm the change from the server
      mutate('triageTickets');
    } catch (err) {
      console.error("Failed to resolve ticket:", err);
      alert("Could not resolve the ticket. Please try again.");
      // Rollback optimistic update on error
      mutate('triageTickets');
    }
  };

  // FIX: Refactor to use makeRequest via apiService for authenticated fetch
  const handleViewPhoto = useCallback(async (mediaId: string) => {
    setIsImageLoading(true);
    setSelectedImage(null); // Clear previous image
    const photoUrl = getTriageMediaUrl(mediaId); // Gets the URL

    try {
        // Use fetch directly for blob response, but get token first
        const token = localStorage.getItem('feelori_admin_token');
        if (!token) throw new Error('Authentication token not found.');

        const response = await fetch(photoUrl, {
            headers: {
            'Authorization': `Bearer ${token}`
            }
        });

      if (!response.ok) {
          let errorMsg = `Failed to fetch image. Status: ${response.status}`;
          try {
              const errBody = await response.json();
              errorMsg = errBody.detail || errorMsg;
          } catch (_) { /* Ignore if body isn't JSON */ }
          throw new Error(errorMsg);
      }

      const blob = await response.blob();
      // Check if blob has size and valid type
       if (blob.size === 0 || !blob.type.startsWith('image/')) {
           throw new Error('Received invalid image data.');
       }
      const imageUrl = URL.createObjectURL(blob);
      setSelectedImage(imageUrl);
    } catch (err) {
      console.error("Failed to load image:", err);
      alert(`Could not load the image: ${err instanceof Error ? err.message : 'Unknown error'}. Please try again.`);
      setSelectedImage(null); // Ensure no broken image is shown
    } finally {
        setIsImageLoading(false);
    }
  }, []); // useCallback dependency array is empty as it doesn't depend on component state/props

  const closeImageModal = useCallback(() => {
    if (selectedImage) {
      URL.revokeObjectURL(selectedImage);
      setSelectedImage(null);
    }
  }, [selectedImage]); // Depend on selectedImage

  const renderContent = () => {
    if (isLoadingTickets) {
      return <div className="text-center p-8">Loading tickets...</div>;
    }

    if (ticketsError) {
      console.error('Triage page error:', ticketsError);
      return (
        <div className="text-center p-8 text-red-600">
          <p>Failed to load tickets.</p>
          <p className="text-sm mt-2">Error: {ticketsError.message}</p>
          <p className="text-xs mt-1 text-gray-500">Check console for more details</p>
        </div>
      );
    }

    if (!data || !data.tickets || data.tickets.length === 0) {
      return <div className="text-center p-8 text-gray-500">No pending triage tickets. All caught up! 🎉</div>;
    }

    return (
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 border border-gray-200">
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
              <tr key={ticket._id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{formatDateTime(ticket.created_at)}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{ticket.order_number}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{ticket.customer_phone}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{ticket.issue_type}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium space-x-2">
                  {ticket.image_media_id && ticket.image_media_id !== 'N/A' && (
                    <Button
                      variant="secondary"
                      size="sm" // Smaller button size
                      onClick={() => handleViewPhoto(ticket.image_media_id!)}
                      disabled={isImageLoading} // Disable while any image is loading
                    >
                      {/* Show loading state specifically for this button if needed, complex */}
                      View Photo
                    </Button>
                  )}
                  <Button
                    variant="primary"
                    size="sm" // Smaller button size
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

  // Add keydown listener for Escape key to close modal
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeImageModal();
      }
    };

    if (selectedImage || isImageLoading) {
      window.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedImage, isImageLoading, closeImageModal]);


  return (
    <div className="p-4 md:p-8 space-y-6">
      <h1 className="text-3xl font-bold tracking-tight text-black">Triage Tickets</h1>
      <Card>
        <CardHeader>
          <CardTitle>Pending Customer Issues</CardTitle>
        </CardHeader>
        <CardContent>
          {renderContent()}
        </CardContent>
      </Card>

      {/* Image Modal */}
      {(selectedImage || isImageLoading) && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={closeImageModal} // Close when clicking backdrop
          aria-labelledby="image-modal-title"
          role="dialog"
          aria-modal="true"
        >
          <div
             className="relative bg-white p-4 rounded-lg shadow-xl max-w-4xl max-h-[90vh] flex items-center justify-center"
             onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside content
          >
            <button
              onClick={closeImageModal}
              className="absolute top-2 right-2 bg-gray-200 rounded-full w-8 h-8 flex items-center justify-center hover:bg-gray-300 text-gray-700 font-bold z-10"
              aria-label="Close image viewer"
            >
              ✕
            </button>
            {isImageLoading && <div className="spinner"></div>}
            {/* FIX: Disable next/no-img-element warning for this specific image */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            {selectedImage && !isImageLoading && (
                 <img
                    src={selectedImage}
                    alt="Ticket photo"
                    className="max-w-full max-h-[85vh] object-contain rounded" // Added rounded corners
                 />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Add Button size prop if not already present
declare module '../../components/ui/Button' {
    interface CustomButtonProps {
        size?: 'sm' | 'default' | 'lg'; // Example sizes
    }
}