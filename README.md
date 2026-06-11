# VoiceDoc Intelligence

Voice-Commanded Document Intelligence Multi-Agent System

## Google Cloud Rapid Agent Hackathon Project

### Tech Stack
- **Primary LLM**: Gemini (gemini-2.0-flash)
- **Orchestration**: Google Cloud Agent Builder + LangGraph
- **Partner Integration**: MongoDB MCP Server (mandatory)
- **Backend**: FastAPI + Celery + Redis
- **Database**: MongoDB Atlas (Motor async driver)
- **Frontend**: HTML + Tailwind + Vanilla JS + Web Speech API
- **Deployment**: Docker + Google Cloud Run

### Architecture

```
User Voice Input → Web Speech API → FastAPI
                                      ↓
                              SupervisorAgent (LangGraph)
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
              ScannerAgent    EvaluatorAgent    ExtractorAgent
                    ↓                 ↓                 ↓
                    └─────────────────┼─────────────────┘
                                      ↓
                              ProcessorAgent
                                      ↓
                              MongoDB Atlas
                                      ↓
                              QueryAgent (RAG)
```

### Features
- 🎤 Voice-commanded document processing
- 🔄 Real-time progress updates via WebSocket
- 🤖 Multi-agent parallel execution
- 🧠 RAG-based question answering
- 📊 Vector search with MongoDB Atlas
- 🚀 Scalable with Celery workers

### Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Start services with Docker Compose
docker-compose up -d

# Run the application
uvicorn app.main:app --reload
```

### MongoDB Collections
- `documents`: Raw documents, metadata, source URL
- `chunks`: Text chunks with vector embeddings
- `agent_memory`: Conversation history per session
- `job_status`: Parallel job tracking
- `query_logs`: Query analytics

### Environment Variables
See `.env.example` for required configuration.

### Build Progress
- [x] Step 1: Project structure + config files
- [ ] Step 2: MongoDB connection + schema
- [ ] Step 3: Redis + Celery configuration
- [ ] Step 4-8: Agent implementation
- [ ] Step 9: RAG pipeline
- [ ] Step 10: FastAPI routes + WebSocket
- [ ] Step 11: MongoDB MCP integration
- [ ] Step 12: Frontend
- [ ] Step 13: Google Cloud Agent Builder
- [ ] Step 14: Deployment
