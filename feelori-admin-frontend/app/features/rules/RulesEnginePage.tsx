import React from 'react';
import { apiService, Rule } from '../../../lib/api';
import { Button } from '../../components/ui/Button';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';
import { Input } from '../../components/ui/Input';
import { TagInput } from '../../components/ui/TagInput';

export const RulesEnginePage = () => {
  const [intents, setIntents] = React.useState<Rule[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    const fetchRules = async () => {
        try {
            setLoading(true);
            const data = await apiService.getRules();
            setIntents(data);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    fetchRules();
  }, []);

  const [keywords, setKeywords] = React.useState(["ruby", "necklace", "earring", "bangle", "bracelet", "ring", "pendant", "gold", "silver", "diamond"]);
  const [testInput, setTestInput] = React.useState("");
  const [testResult, setTestResult] = React.useState<{message: string, type: string} | null>(null);

  const [isSaving, setIsSaving] = React.useState(false);

  const handleAddIntent = async () => {
    const newRule = { name: "new_intent", keywords: [], phrases: [] };
    try {
        const createdRule = await apiService.createRule(newRule);
        setIntents([...intents, createdRule]);
    } catch (err: any) {
        setError(err.message);
    }
  };

  const handleIntentChange = async (index: number, field: string, value: any) => {
    const updatedIntents = [...intents];
    const updatedIntent = { ...updatedIntents[index], [field]: value };
    updatedIntents[index] = updatedIntent;
    setIntents(updatedIntents);

    try {
        setIsSaving(true);
        await apiService.updateRule(updatedIntent._id!, updatedIntent);
    } catch (err: any) {
        setError(err.message);
        // Optionally revert the change here
    } finally {
        setIsSaving(false);
    }
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

  if (loading) return <div className="text-center p-10">Loading rules...</div>;
  if (error) return <div className="text-center p-10 text-red-500">Error: {error}</div>;

  return (
    <div className="space-y-6">
        <div className="flex items-center justify-between">
            <div>
                <h1 className="text-3xl font-bold text-black">Rules Engine</h1>
                <p className="text-gray-700 mt-1">Manage AI intents, keywords, and test matching logic.</p>
            </div>
            {isSaving && <div className="text-sm text-gray-500">Saving...</div>}
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
                            <div key={intent._id || `new-intent-${idx}`} className="p-4 border rounded-lg bg-gray-50/50">
                                <Input
                                    className="text-md font-semibold mb-3 bg-white"
                                    value={intent.name}
                                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleIntentChange(idx, 'name', e.target.value)}
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
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTestInput(e.target.value)}
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