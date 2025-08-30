"use client";
import React, { useState, useEffect } from 'react';
import { apiService } from '../../../../lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '../../../components/ui/Card';
import { Button } from '../../../components/ui/Button';

interface Message {
  timestamp: string;
  message: string;
  response: string;
}

interface Customer {
  _id: string;
  name: string;
  phone_number: string;
  conversation_history: Message[];
}

const CustomerChatPage = ({ customerId, onBack }: { customerId: string; onBack: () => void; }) => {
  const [customer, setCustomer] = useState<Customer | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!customerId) return;

    const fetchCustomerData = async () => {
      setLoading(true);
      try {
        const response = await apiService.getCustomerById(customerId);
        setCustomer(response.customer);
      } catch (err: any) {
        setError(err.message || 'Failed to fetch customer data.');
      } finally {
        setLoading(false);
      }
    };

    fetchCustomerData();
  }, [customerId]);

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('en-IN', {
      timeZone: 'Asia/Kolkata',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    });
  };

  if (loading) return <div>Loading chat...</div>;
  if (error) return <div className="text-red-500">{error}</div>;
  if (!customer) return <div>Customer not found.</div>;

  return (
    <div className="space-y-6">
       <Button onClick={onBack}>&larr; Back to Conversations</Button>
      <Card>
        <CardHeader>
          <CardTitle>Chat with {customer.name} ({customer.phone_number})</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {customer.conversation_history.map((entry, index) => (
            <React.Fragment key={index}>
              <div className="p-3 bg-blue-100 text-blue-900 rounded-lg max-w-xl ml-auto mb-2">
                <p className="text-sm">{entry.message}</p>
                <p className="text-xs text-right opacity-70 mt-1">{formatTimestamp(entry.timestamp)}</p>
              </div>
              {entry.response && (
                <div className="p-3 bg-gray-100 text-gray-900 rounded-lg max-w-xl mr-auto">
                   <p className="text-sm">{entry.response}</p>
                   <p className="text-xs text-left opacity-70 mt-1">{formatTimestamp(entry.timestamp)}</p>
                </div>
              )}
            </React.Fragment>
          ))}
        </CardContent>
      </Card>
    </div>
  );
};

export default CustomerChatPage;