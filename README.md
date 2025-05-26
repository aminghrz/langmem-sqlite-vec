# langmem-sqlite-vec
Persistent drop-in replacement for LangGraph's InMemoryStore with vector search capabilities using sqlite3 and sqlite-vec to use as store in langmem

# SQLite-Vec Store for LangGraph

> **Persistent drop-in replacement for LangGraph's InMemoryStore with vector search**

Stop losing your agent's memory when your chatbot restarts! 

## The Problem
```python
# This loses all memory when your app restarts 😢
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()
```

# This persists everything to disk + adds vector search 🚀
```python
from sqlite_vec_store import SqliteVecStore
store = SqliteVecStore(db_file="chatbot_memory.db")
# That's it! Same API, persistent storage
```
# Example
```python
from langgraph.checkpoint.sqlite import SqliteSaver, sqlite3
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

# After downloading and moving sqlite_vec_store.py in the same directory as your current script:
from sqlite_vec_store import SqliteVecStore

con = sqlite3.connect("chatbot.sqlite3", check_same_thread=False)

embedding_model = OpenAIEmbeddings(
        model='text-embedding-3-large'
    )
chat_model = ChatOpenAI(
    temperature=0,
    model='gpt-4.1'
)

store = SqliteVecStore(
    db_file='chatbot.sqlite3',
    index={
        "dims": 3072,
        "embed": embedding_model,
    }
)

manage_memory = create_manage_memory_tool(store=store, namespace=("memory","{user_id}"))
search_memory = create_search_memory_tool(store=store, namespace=("memory","{user_id}"))

agent = create_react_agent(
    model=chat_model,
    tools=[manage_memory, search_memory],
    prompt=SystemMessage("You are a helpful assistant. Respond to the user's last message based on the provided context and conversation history and memories. Store any interests and topics the user talks about."),
    checkpointer=SqliteSaver(con),
    store=store,
)
# Creating memory
config = {
    "configurable": {
        "thread_id": "thread_1",
        "user_id": "cat_lover"
    }
}

inputs = {"messages": {"role": "user", "content": "I love cats"}}

response = agent.invoke(inputs, config)
print(response.items().mapping['messages'][-1].content)

# Recalling memory
config = {
    "configurable": {
        "thread_id": "thread_2",
        "user_id": "cat_lover"
    }
}

inputs = {"messages": {"role": "user", "content": "what do i love?"}}
response = agent.invoke(inputs, config)
print(response.items().mapping['messages'][-1].content)

print(store.list_namespaces())
```

# Known Issues
Search method returns empty list. But langmem tools work perfectly.
