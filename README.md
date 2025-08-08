# Langmem SQLite Vector Store
Persistent drop-in replacement for LangGraph's InMemoryStore with vector search capabilities using sqlite3 and sqlite-vec to use as store in langmem and other RAG applications.

# SQLite-Vec Store for LangGraph

> **Persistent drop-in replacement for LangGraph's InMemoryStore with vector search**

Stop losing your agent's memory when your chatbot restarts! 

## The Problem
Currently there are two main options for persistent store:  
1 - Postgres + PGVector  
This has a lot of setup and requires a lot of space. This makes it less portable and compact.  
2 - InMemorySrore  
It stores in RAM anytime anywhere but loses all memory when your app restarts 😢

# This persists everything to disk + adds vector search 🚀
```python
from sqlite_vec_store import SqliteVecStore
store = SqliteVecStore(db_file="chatbot_memory.db")
# or
store = SqliteVecStore(db_file="chatbot_memory.sqlite3")
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
print(store.search(('memory', 'cat_lover')))
```

# Requirements:
asyncio  
concurrent  
json  
logging  
sqlite3  
struct  
collections  
datetime  
typing  
langchain_core  
langgraph  
sqlite-vec
