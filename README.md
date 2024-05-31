# Compas Assistant
Template de OpenAI [Assistants API](https://platform.openai.com/docs/assistants/overview) with [Next.js](https://nextjs.org/docs).
<br/>
<br/>


## Rapide Setup

### 1. Clone repo
```shell
git clone https://github.com/DenezG/compas_assistant_js_ragV4.git
cd compas_assistant_js_ragV4
```

### 2. Configurer votre [OpenAI API key](https://platform.openai.com/api-keys)
```shell
OPENAI_API_KEY = 'sk-proj-...'
ASSISTANT_ID = 'asst_...'
DEWY_ENDPOINT= 'http://localhost:8000'
DEWY_COLLECTION= 'main'
```
(Dans un fichier `.env.local` que vous devez créer).

### 3. Install dependencies
```shell
npm install
```

### 4. Run
```shell
npm run dev
```

### 5. Naviger à [http://localhost:3000](http://localhost:3000).

## Overview

Ce projet représente un assistant Openai qui utilise les données du compas afin de répondre au questions des utilisateurs.
Nous vous conseillons d'utiliser la page 'Chat avec Images' car c'est la plus avancée et elle regroupe les fonctionnalités des deux autres pages.


### Main Components

- `app/components/chat.tsx` - handles chat rendering, [streaming](https://platform.openai.com/docs/assistants/overview?context=with-streaming), and [function call](https://platform.openai.com/docs/assistants/tools/function-calling/quickstart?context=streaming&lang=node.js) forwarding
- `app/components/file-viewer.tsx` - handles uploading, fetching, and deleting files for [file search](https://platform.openai.com/docs/assistants/tools/file-search)

### Endpoints

- `api/assistants` - `POST`: create assistant (only used at startup)
- `api/assistants/threads` - `POST`: create new thread
- `api/assistants/threads/[threadId]/messages` - `POST`: send message to assistant
- `api/assistants/threads/[threadId]/actions` - `POST`: inform assistant of the result of a function it decided to call
- `api/assistants/files` - `GET`/`POST`/`DELETE`: fetch, upload, and delete assistant files for file search
- `api/assistants/image` - `POST` : fetch an image by using the fileId of the image


# RAG Setup

Api Key: https://unstructured.io/api-key-free
