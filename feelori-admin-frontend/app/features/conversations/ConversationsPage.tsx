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
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message || 'Failed to fetch conversations.');
      } else {
        setError('An unknown error occurred.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCustomers();
  }, []);

  const handlePageChange = (newPage: number) => {
    if (newPage > 0 && newPage <= pagination.pages) {
      fetchCustomers(newPage);
    }
  };


  if (loading) return <div className="text-center p-8">Loading conversations...</div>;
  if (error) return <div className="text-center p-8 text-red-600">Error: {error}</div>;

  return (
    <div className="space-y-6">
       <Card>
        <CardHeader>
          <CardTitle>All Conversations</CardTitle>
          <CardDescription>
            Showing {customers.length} of {pagination.total} conversations.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Customer</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Interaction</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {customers.map((customer) => (
                  <tr key={customer._id}>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{customer.name}</div>
                      <div className="text-sm text-gray-500">{customer.phone_number}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(customer.last_interaction).toLocaleString('en-IN', {
                        dateStyle: 'short',
                        timeStyle: 'medium',
                        timeZone: 'Asia/Kolkata',
                      })}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
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