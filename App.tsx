
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Chat, Modality, LiveSession, LiveServerMessage } from '@google/genai';

// --- TYPE DEFINITIONS ---
interface ChatMessage {
  role: 'user' | 'model';
  content: string;
}

type LiveState = 'idle' | 'connecting' | 'active' | 'error';


// --- AUDIO HELPER FUNCTIONS ---
// Decodes a base64 string into a Uint8Array.
function decode(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

// Encodes a Uint8Array into a base64 string.
function encode(bytes: Uint8Array): string {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

// Decodes raw PCM audio data into an AudioBuffer for playback.
async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

// Creates a Blob-like object for sending audio data to the API.
function createBlob(data: Float32Array): { data: string; mimeType: string; } {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}


// --- SVG ICONS (HELPER COMPONENTS) ---
const BotIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-white bg-green-500 rounded-full p-1" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2a10 10 0 0 0-9.99 9h2.02A8 8 0 0 1 12 4v8H4a8 8 0 0 1 1.48-4.52L3.5 6.5A10 10 0 0 0 2 12c0 5.52 4.48 10 10 10s10-4.48 10-10S17.52 2 12 2zM6.5 17.5l1.48-1.48A8 8 0 0 1 4 12H2a10 10 0 0 0 9.99 9z" />
    <path d="M12 20a8 8 0 0 1-7.98-7h-2.02A10 10 0 0 0 12 22v-2z" />
    <path d="M17.5 6.5L16.02 7.98A8 8 0 0 1 20 12h2a10 10 0 0 0-9.99-9z" />
  </svg>
);

const UserIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-white bg-blue-500 rounded-full p-1.5" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
  </svg>
);

const SendIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 24 24" fill="currentColor">
    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
  </svg>
);

const MicrophoneIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.49 6-3.31 6-6.72h-1.7z" />
    </svg>
);

const ClearIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M17.65 6.35A7.958 7.958 0 0 0 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
    </svg>
);

// --- UI COMPONENTS ---
const LoadingIndicator = () => (
  <div className="flex items-center space-x-2 justify-start">
    <BotIcon />
    <div className="flex space-x-1">
      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
    </div>
  </div>
);

interface MessageProps {
  message: ChatMessage;
  isPartial?: boolean;
}

const Message: React.FC<MessageProps> = ({ message, isPartial = false }) => {
  const isModel = message.role === 'model';
  return (
    <div className={`flex items-start space-x-4 ${isModel ? 'justify-start' : 'justify-end'} ${isPartial ? 'opacity-70' : ''}`}>
      {isModel && <BotIcon />}
      <div className={`px-4 py-3 rounded-2xl max-w-lg whitespace-pre-wrap ${isModel ? 'bg-gray-700 text-gray-200 rounded-tl-none' : 'bg-blue-600 text-white rounded-br-none'}`}>
        {message.content}
      </div>
      {!isModel && <UserIcon />}
    </div>
  );
};

// --- MAIN APP COMPONENT ---
const App: React.FC = () => {
  const [chat, setChat] = useState<Chat | null>(null);
  const [history, setHistory] = useState<ChatMessage[]>([
    {
      role: 'model',
      content: 'Olá! Sou seu tutor especialista em Google Workspace for Education. Como posso ajudar você hoje com as ferramentas Google ou Chromebooks?',
    },
  ]);
  const [userInput, setUserInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  
  // Live API State
  const [liveState, setLiveState] = useState<LiveState>('idle');
  const [currentInputTranscription, setCurrentInputTranscription] = useState('');
  const [currentOutputTranscription, setCurrentOutputTranscription] = useState('');

  // Refs
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const liveSessionRef = useRef<LiveSession | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const audioSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const nextStartTimeRef = useRef<number>(0);


  const SYSTEM_INSTRUCTION = "Você é um professor tutor especialista nas ferramentas Google Workspace for Education. Você possui as certificações Nível 1, Nível 2, Trainer, Coach e Innovator do Google. Sua missão é ajudar outros professores a tirar dúvidas sobre como utilizar as ferramentas do Google Workspace e Chromebooks. Responda de forma simpática, gentil e didática, simulando o comportamento humano para criar uma conversa agradável e acolhedora. Use formatação como quebras de linha para melhor legibilidade.";

  const initializeChat = useCallback(() => {
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
      const chatInstance = ai.chats.create({
        model: 'gemini-2.5-flash',
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
        },
      });
      setChat(chatInstance);
    } catch (error) {
      console.error("Failed to initialize Gemini:", error);
      setHistory(prev => [...prev, { role: 'model', content: 'Desculpe, não consegui me conectar ao serviço de IA. Verifique a configuração da sua API Key.' }]);
    }
  }, [SYSTEM_INSTRUCTION]);

  useEffect(() => {
    initializeChat();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    chatContainerRef.current?.scrollTo({
      top: chatContainerRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [history, isLoading, currentInputTranscription, currentOutputTranscription]);
  
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading || !chat || liveState !== 'idle') return;

    const userMessage: ChatMessage = { role: 'user', content: userInput.trim() };
    setHistory(prev => [...prev, userMessage]);
    setUserInput('');
    setIsLoading(true);

    try {
      const response = await chat.sendMessage({ message: userMessage.content });
      const modelMessage: ChatMessage = { role: 'model', content: response.text };
      setHistory(prev => [...prev, modelMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: ChatMessage = {
        role: 'model',
        content: 'Ocorreu um erro ao processar sua solicitação. Por favor, tente novamente.',
      };
      setHistory(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const cleanupLiveSession = useCallback(() => {
    liveSessionRef.current?.close();
    liveSessionRef.current = null;
    
    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    mediaStreamRef.current = null;

    scriptProcessorRef.current?.disconnect();
    scriptProcessorRef.current = null;
    
    inputAudioContextRef.current?.close();
    outputAudioContextRef.current?.close();
    
    audioSourcesRef.current.forEach(source => source.stop());
    audioSourcesRef.current.clear();
    
    setLiveState('idle');
    setCurrentInputTranscription('');
    setCurrentOutputTranscription('');
}, []);


  const toggleLiveSession = useCallback(async () => {
    if (liveState === 'active' || liveState === 'connecting') {
        cleanupLiveSession();
        return;
    }
    
    try {
        setLiveState('connecting');
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaStreamRef.current = stream;

        // Fix for: Property 'webkitAudioContext' does not exist on type 'Window & typeof globalThis'.
        inputAudioContextRef.current = new ((window as any).AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
        // Fix for: Property 'webkitAudioContext' does not exist on type 'Window & typeof globalThis'.
        outputAudioContextRef.current = new ((window as any).AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        nextStartTimeRef.current = 0;

        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
        const sessionPromise = ai.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-09-2025',
            config: {
                responseModalities: [Modality.AUDIO],
                inputAudioTranscription: {},
                outputAudioTranscription: {},
                systemInstruction: SYSTEM_INSTRUCTION,
            },
            callbacks: {
                onopen: () => {
                    setLiveState('active');
                    const source = inputAudioContextRef.current!.createMediaStreamSource(mediaStreamRef.current!);
                    const scriptProcessor = inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
                    scriptProcessorRef.current = scriptProcessor;

                    scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
                        const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                        const pcmBlob = createBlob(inputData);
                        sessionPromise.then((session) => {
                            session.sendRealtimeInput({ media: pcmBlob });
                        });
                    };
                    source.connect(scriptProcessor);
                    scriptProcessor.connect(inputAudioContextRef.current!.destination);
                },
                onmessage: async (message: LiveServerMessage) => {
                    if (message.serverContent?.inputTranscription) {
                        setCurrentInputTranscription(message.serverContent.inputTranscription.text);
                    }
                    if (message.serverContent?.outputTranscription) {
                        setCurrentOutputTranscription(message.serverContent.outputTranscription.text);
                    }
                    if (message.serverContent?.turnComplete) {
                        const finalInput = currentInputTranscription + (message.serverContent.inputTranscription?.text ?? '');
                        const finalOutput = currentOutputTranscription + (message.serverContent.outputTranscription?.text ?? '');
                        if (finalInput) {
                            setHistory(prev => [...prev, {role: 'user', content: finalInput}]);
                        }
                        if(finalOutput){
                            setHistory(prev => [...prev, {role: 'model', content: finalOutput}]);
                        }
                        setCurrentInputTranscription('');
                        setCurrentOutputTranscription('');
                    }

                    const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                    if (base64Audio) {
                        const outputContext = outputAudioContextRef.current!;
                        nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputContext.currentTime);
                        const audioBuffer = await decodeAudioData(decode(base64Audio), outputContext, 24000, 1);
                        const source = outputContext.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(outputContext.destination);
                        source.addEventListener('ended', () => audioSourcesRef.current.delete(source));
                        source.start(nextStartTimeRef.current);
                        nextStartTimeRef.current += audioBuffer.duration;
                        audioSourcesRef.current.add(source);
                    }
                    if (message.serverContent?.interrupted) {
                        audioSourcesRef.current.forEach(source => source.stop());
                        audioSourcesRef.current.clear();
                        nextStartTimeRef.current = 0;
                    }
                },
                onerror: (e: ErrorEvent) => {
                    console.error("Live session error:", e);
                    setHistory(prev => [...prev, {role: 'model', content: 'Desculpe, a conexão de voz falhou.'}]);
                    cleanupLiveSession();
                },
                onclose: () => {
                    cleanupLiveSession();
                },
            },
        });

        sessionPromise.then(session => {
            liveSessionRef.current = session;
        }).catch(e => {
            console.error("Failed to connect live session:", e);
            setHistory(prev => [...prev, {role: 'model', content: 'Não foi possível iniciar a sessão de voz.'}]);
            cleanupLiveSession();
        });

    } catch (error) {
        console.error('Failed to get user media:', error);
        setHistory(prev => [...prev, {role: 'model', content: 'Permissão para microfone negada. Por favor, habilite o microfone para usar o chat de voz.'}]);
        cleanupLiveSession();
    }
}, [liveState, cleanupLiveSession, SYSTEM_INSTRUCTION, currentInputTranscription, currentOutputTranscription]);

  useEffect(() => {
    return () => cleanupLiveSession();
  }, [cleanupLiveSession]);

  const handleClearChat = () => {
      if (isLoading) setIsLoading(false);
      if (liveState !== 'idle') cleanupLiveSession();
      setHistory([
        {
          role: 'model',
          content: 'Olá! Sou seu tutor especialista em Google Workspace for Education. Como posso ajudar você hoje com as ferramentas Google ou Chromebooks?',
        },
      ]);
  };

  const isMicActive = liveState === 'active' || liveState === 'connecting';

  return (
    <div className="flex flex-col h-screen bg-gray-800 text-white font-sans">
      <header className="bg-gray-900/80 backdrop-blur-sm shadow-md p-4 border-b border-gray-700 flex justify-between items-center">
        <h1 className="text-xl md:text-2xl font-bold text-center text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-blue-500 flex-1">
          Tutor Google Workspace for Education
        </h1>
        <button onClick={handleClearChat} title="Clear Chat" className="text-gray-400 hover:text-white transition-colors duration-200">
            <ClearIcon />
        </button>
      </header>
      
      <main ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
        {history.map((msg, index) => (
          <Message key={index} message={msg} />
        ))}
        {isLoading && <LoadingIndicator />}
        {currentInputTranscription && <Message message={{role: 'user', content: currentInputTranscription}} isPartial={true} />}
        {currentOutputTranscription && <Message message={{role: 'model', content: currentOutputTranscription}} isPartial={true} />}

      </main>
      
      <footer className="p-4 bg-gray-900/80 backdrop-blur-sm border-t border-gray-700">
        <form onSubmit={handleSendMessage} className="flex items-center space-x-2 md:space-x-4 max-w-4xl mx-auto">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder={isMicActive ? "Ouvindo..." : "Pergunte sobre as ferramentas Google..."}
            className="flex-1 bg-gray-700 border border-gray-600 rounded-full py-3 px-5 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-200 disabled:opacity-50"
            disabled={isLoading || isMicActive}
          />
          <button
            type="button"
            onClick={toggleLiveSession}
            title={isMicActive ? "Stop voice chat" : "Start voice chat"}
            className={`rounded-full p-3 transition duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 ${
                isMicActive ? 'bg-red-600 hover:bg-red-700 text-white animate-pulse' : 'bg-green-600 hover:bg-green-700 text-white'
            } ${liveState === 'connecting' ? 'cursor-wait' : ''}`}
          >
            <MicrophoneIcon />
          </button>
          <button
            type="submit"
            disabled={isLoading || !userInput.trim() || isMicActive}
            className="bg-blue-600 text-white rounded-full p-3 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-blue-500"
          >
            <SendIcon />
          </button>
        </form>
      </footer>
    </div>
  );
};

export default App;
