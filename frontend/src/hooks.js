import { useCallback, useEffect, useState } from "react";

const uid = () => Math.random().toString(36).slice(2, 10);

// Chats live in this browser only (localStorage), separately for each signed-in user.
export function useChats(userKey) {
  const storageKey = `hunter-chats:${userKey}`;
  const [chats, setChats] = useState([]);
  const [activeId, setActiveId] = useState(null);

  useEffect(() => {
    let saved = [];
    try {
      saved = JSON.parse(localStorage.getItem(storageKey) || "[]");
    } catch {
      /* corrupt or blocked */
    }
    setChats(saved);
    setActiveId(saved[0]?.id ?? null);
  }, [storageKey]);

  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(chats.slice(0, 50)));
    } catch {
      /* quota or blocked */
    }
  }, [chats, storageKey]);

  const newChat = useCallback(() => setActiveId(null), []);

  const addMessage = useCallback((chatId, message) => {
    setChats((prev) => prev.map((c) => (c.id === chatId ? { ...c, messages: [...c.messages, message] } : c)));
  }, []);

  const startChat = useCallback((firstMessage) => {
    const chat = {
      id: uid(),
      title: firstMessage.content.slice(0, 48),
      messages: [firstMessage],
    };
    setChats((prev) => [chat, ...prev]);
    setActiveId(chat.id);
    return chat.id;
  }, []);

  const removeChat = useCallback((id) => {
    setChats((prev) => prev.filter((c) => c.id !== id));
    setActiveId((cur) => (cur === id ? null : cur));
  }, []);

  return { chats, activeId, setActiveId, newChat, startChat, addMessage, removeChat };
}
