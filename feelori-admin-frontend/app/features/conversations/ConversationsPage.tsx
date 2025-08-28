import React from 'react';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';

export const ConversationsPage = () => {
    const [conversations, setConversations] = React.useState<any[]>([]);
    const [selectedConvo, setSelectedConvo] = React.useState<any>(null);
    const [messages, setMessages] = React.useState<any[]>([]);
    const [loading, setLoading] = React.useState(true);

    React.useEffect(() => {
        setTimeout(() => {
            setConversations([
                { id: 1, name: 'Priya Sharma', phone: '+919876543210', lastMessage: 'Is this available in gold?', time: '5m ago', status: 'delivered' },
                { id: 2, name: 'Amit Singh', phone: '+919123456789', lastMessage: 'Thank you!', time: '2h ago', status: 'read' },
                { id: 3, name: 'Sneha Reddy', phone: '+918765432109', lastMessage: 'What is the price for #1234?', time: '1d ago', status: 'sent' },
            ]);
            setLoading(false);
        }, 1000);
    }, []);

    const handleSelectConvo = (convo: any) => {
        setSelectedConvo(convo);
        setMessages([
            { from: 'user', text: 'Hi, I saw a necklace I liked.' },
            { from: 'bot', text: 'Hello! I can help with that. Could you describe it or send a picture?' },
            { from: 'user', text: 'It was a gold choker with red stones.' },
            { from: 'bot', text: 'Searching for gold chokers with red stones for you now... ✨' },
        ]);
    };

    const StatusTick = ({ status }: { status: string }) => {
        if (status === 'read') return <span className="text-blue-500">✓✓</span>;
        if (status === 'delivered') return <span className="text-gray-900">✓✓</span>;
        return <span className="text-gray-900">✓</span>;
    };

    return (
        <div className="h-[calc(100vh-84px)] flex flex-col">
            <h1 className="text-3xl font-bold text-black pb-4">Conversation Explorer</h1>
            <div className="flex-grow grid grid-cols-1 lg:grid-cols-3 gap-6 overflow-hidden">
                <Card className="lg:col-span-1 flex flex-col">
                    <CardHeader><CardTitle>Recent Chats</CardTitle></CardHeader>
                    <CardContent className="flex-grow overflow-y-auto p-2">
                        {loading ? <p className="text-gray-900">Loading...</p> : (
                            <div className="space-y-2">
                                {conversations.map(convo => (
                                    <div key={convo.id} onClick={() => handleSelectConvo(convo)}
                                        className={`p-3 rounded-lg cursor-pointer transition-colors ${selectedConvo?.id === convo.id ? 'bg-[#ff4d6d] text-white' : 'hover:bg-gray-100'}`}>
                                        <div className="flex justify-between items-center">
                                            <p className={`font-semibold ${selectedConvo?.id === convo.id ? 'text-white' : 'text-gray-900'}`}>{convo.name}</p>
                                            <p className={`text-xs ${selectedConvo?.id === convo.id ? 'text-gray-200' : 'text-gray-700'}`}>{convo.time}</p>
                                        </div>
                                        <div className="flex justify-between items-center">
                                            <p className={`text-sm truncate ${selectedConvo?.id === convo.id ? 'text-gray-200' : 'text-gray-700'}`}>{convo.lastMessage}</p>
                                            <StatusTick status={convo.status} />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </CardContent>
                </Card>
                <Card className="lg:col-span-2 flex flex-col">
                    <CardHeader>
                        <CardTitle>{selectedConvo ? selectedConvo.name : 'Select a Conversation'}</CardTitle>
                        <CardDescription>{selectedConvo ? selectedConvo.phone : 'No chat selected'}</CardDescription>
                    </CardHeader>
                    <CardContent className="flex-grow overflow-y-auto bg-gray-50">
                        {selectedConvo ? (
                            <div className="space-y-4 text-gray-900">
                                {messages.map((msg, i) => (
                                    <div key={i} className={`flex ${msg.from === 'user' ? 'justify-end' : 'justify-start'}`}>
                                        <div className={`max-w-xs lg:max-w-md p-3 rounded-lg ${msg.from === 'user' ? 'bg-[#ff4d6d] text-white' : 'bg-white border'}`}>
                                            {msg.text}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="flex items-center justify-center h-full text-gray-700">
                                <p>Select a conversation from the left to view the transcript.</p>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};
