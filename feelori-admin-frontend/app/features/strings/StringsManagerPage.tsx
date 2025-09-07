import React from 'react';
import { apiService, StringResource } from '../../../lib/api';
import { Button } from '../../components/ui/Button';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';
import { Textarea } from '../../components/ui/Textarea';

export const StringsManagerPage = () => {
    const [strings, setStrings] = React.useState<Record<string, string>>({});
    const [loading, setLoading] = React.useState(true);
    const [isSaving, setIsSaving] = React.useState(false);
    const [error, setError] = React.useState<string | null>(null);

    const fetchStrings = async () => {
        try {
            setLoading(true);
            setError(null);
            const data = await apiService.getStrings();
            const stringsMap = data.reduce((acc, s) => ({ ...acc, [s.key]: s.value }), {});
            setStrings(stringsMap);
        } catch (err) {
            if (err instanceof Error) {
                setError(err.message);
            } else {
                setError('An unknown error occurred while fetching strings.');
            }
        } finally {
            setLoading(false);
        }
    };

    React.useEffect(() => {
        fetchStrings();
    }, []);

    const handleSave = async () => {
        try {
            setIsSaving(true);
            setError(null);
            const stringsToSave = Object.entries(strings).map(([key, value]) => ({ key, value }));
            await apiService.updateStrings(stringsToSave);
        } catch (err) {
            if (err instanceof Error) {
                setError(err.message);
            } else {
                setError('An unknown error occurred while saving strings.');
            }
        } finally {
            setIsSaving(false);
        }
    };

    const handleStringChange = (key: string, value: string) => {
        setStrings(prev => ({ ...prev, [key]: value }));
    };

    if (loading) return <div className="text-center p-8">Loading strings...</div>;
    if (error) return <div className="text-center p-8 text-red-600">Error: {error}</div>;

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-black">Strings Manager</h1>
                    <p className="text-gray-700 mt-1">Edit all user-facing text and messages from one place.</p>
                </div>
                <div className="flex gap-2">
                    <Button onClick={handleSave} variant="primary" disabled={isSaving}>
                        {isSaving ? 'Saving...' : 'Save Changes'}
                    </Button>
                    <Button onClick={fetchStrings} variant="secondary" disabled={loading}>Rollback</Button>
                </div>
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>String Resources</CardTitle>
                    <CardDescription>Update all user-facing text and messages.</CardDescription>
                </CardHeader>
                <CardContent className="grid md:grid-cols-2 gap-6">
                    {Object.entries(strings).map(([key, value]) => (
                        <div key={key}>
                            <label className="text-sm font-medium text-gray-700 mb-2 block">{key}</label>
                            <Textarea
                                rows={10}
                                value={value}
                                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => handleStringChange(key, e.target.value)}
                            />
                        </div>
                    ))}
                </CardContent>
            </Card>
        </div>
    );
};