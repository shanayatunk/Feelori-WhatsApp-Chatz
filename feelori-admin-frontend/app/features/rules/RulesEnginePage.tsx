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

  const fetchRules = async () => {
      try {
          setLoading(true);
          const data = await apiService.getRules();
          setIntents(data);
      } catch (err) {
          // This is the only section that has been modified to fix the build error
          if (err instanceof Error) {
              setError(err.message);
          } else {
              setError('An unknown error occurred.');
          }
      } finally {
          setLoading(false);
      }
  };

  React.useEffect(() => {
    fetchRules();
  }, []);

  const [keywords, setKeywords] = React.useState(["ruby", "necklace", "earring", "bangle", "bracelet", "ring", "pendant", "gold", "silver", "diamond"]);
  const [testInput, setTestInput] = React.useState("");
  const [testResult, setTestResult] = React.useState<{message: string, type: string} | null>(null);

  const [isSaving, setIsSaving] = React.useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    try {
        await Promise.all(intents.map(intent =>
            apiService.updateRule(intent._id!, intent)
        ));
        // You might want a success message here
    } catch (err) {
        if (err instanceof Error) {
            setError(err.message);
        } else {
            setError('An unknown error occurred while saving.');
        }
    } finally {
        setIsSaving(false);
    }
  };

  const handleTest = () => {
    const input = testInput.toLowerCase().trim();
    if (!input) {
        setTestResult({ message: "Please enter a message to test.", type: 'error'});
        return;
    }

    let matchedIntent = 'default_fallback';
    for (const intent of intents) {
        const hasKeyword = intent.keywords.some(k => input.includes(k.toLowerCase()));
        if (hasKeyword) {
            matchedIntent = intent.name;
            break;
        }
    }
    setTestResult({ message: `Matched Intent: ${matchedIntent}`, type: 'success' });
  };

  if (loading) return <div className="text-center p-8">Loading rules...</div>;
  if (error) return <div className="text-center p-8 text-red-600">Error: {error}</div>;

  return (
    <div className="space-y-6">
        <div className="flex justify-between items-center">
            <h1 className="text-3xl font-bold text-black">Rules Engine</h1>
            <Button onClick={handleSave} disabled={isSaving}>
                {isSaving ? 'Saving...' : 'Save All Changes'}
            </Button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
                 {intents.map((intent) => (
                    <Card key={intent.name}>
                        <CardHeader>
                            <CardTitle className="capitalize">{intent.name.replace(/_/g, ' ')}</CardTitle>
                            <CardDescription>Triggers when a message contains these keywords.</CardDescription>
                        </CardHeader>
                        <CardContent>
                           <TagInput tags={intent.keywords} setTags={(newTags) => {
                               setIntents(intents.map(i => i.name === intent.name ? {...i, keywords: newTags} : i))
                           }} placeholder="Add keyword..."/>
                        </CardContent>
                    </Card>
                ))}
            </div>
             <div className="lg:col-span-1 space-y-6">
                <Card>
                    <CardHeader>
                        <CardTitle>Test Intent Matching</CardTitle>
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