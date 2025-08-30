"use client";
import React, { useState, useEffect } from 'react';
import { apiService } from '../../../lib/api';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';

// Define a type for our customer data for better type safety
interface Customer {
  _id: string;
  phone_number: string;
  name: string;
  last_interaction: string;
}

// The component now accepts an `onViewCustomer` function as a prop
export const ConversationsPage = ({ onViewCustomer }: { onViewCustomer: (id: string) => void }) => {
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pagination, setPagination] = useState({ page: 1, limit: 15, total: 0, pages: 1 });

  const fetchCustomers = async (page = 1) => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.getCustomers(page, pagination.limit);
      setCustomers(response.customers);
      setPagination(response.pagination);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch conversations.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCustomers(pagination.page);
  }, [pagination.page]);

  const handlePageChange = (newPage: number) => {
    if (newPage > 0 && newPage <= pagination.pages) {
      setPagination(prev => ({ ...prev, page: newPage }));
    }
  };
  
  if (loading) {
    return <div className="text-center p-10">Loading conversations...</div>;
  }

  if (error) {
    return <div className="text-center p-10 text-red-500">{error}</div>;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-black">Customer Conversations</h1>
      <Card>
        <CardHeader>
          <CardTitle>Recent Interactions</CardTitle>
          <CardDescription>
            Showing {customers.length} customers on page {pagination.page} of {pagination.pages}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="border rounded-lg">
            <table className="min-w-full divide-y">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Phone Number</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Interaction</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y">
                {customers.map((customer) => (
                  <tr key={customer._id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{customer.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{customer.phone_number}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {/* FIX: Be more explicit with date and time formatting options for better browser compatibility */}
                      {new Date(customer.last_interaction).toLocaleString('en-IN', {
                        dateStyle: 'short',
                        timeStyle: 'medium',
                        timeZone: 'Asia/Kolkata',
                      })}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      {/* This button now calls the onViewCustomer function passed from the parent */}
                      <Button className="text-sm" onClick={() => onViewCustomer(customer._id)}>
                        View Chat
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex justify-between items-center mt-4">
              <Button onClick={() => handlePageChange(pagination.page - 1)} disabled={pagination.page <= 1}>
                Previous
              </Button>
              <span className="text-sm">
                Page {pagination.page} of {pagination.pages}
              </span>
              <Button onClick={() => handlePageChange(pagination.page + 1)} disabled={pagination.page >= pagination.pages}>
                Next
              </Button>
            </div>
        </CardContent>
      </Card>
    </div>
  );
};
