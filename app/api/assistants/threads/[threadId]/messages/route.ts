import { assistantId } from "@/app/assistant-config";
import { openai } from "@/app/openai";
import { UnstructuredDirectoryLoader } from "@langchain/community/document_loaders/fs/unstructured";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { OpenAIEmbeddings } from "@langchain/openai";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";

export const runtime = "nodejs";


const options = {
  apiKey: process.env.UNSTRUCTURED_API_KEY,
};



async function getEmbeddingFunction() {
  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text", //A définir pour un embedding spécifique 
  });
  //const embeddings = new OpenAIEmbeddings(); //Chroma ne supporte pas hunggingface sur js
  return embeddings;
}

//RAG model
async function Context(query:string) {
  const loader = new UnstructuredDirectoryLoader(
    "datas/COMPAS",
    options
  );
  const docs = await loader.load();

  //console.log(docs[0].pageContent);
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const embeddings = await getEmbeddingFunction();

  const splits = await textSplitter.splitDocuments(docs);
  /**Pour vectorStore il n'y a pas DuckDB en javascript 
   * Version MemoryVectorStore: https://js.langchain.com/v0.1/docs/modules/data_connection/vectorstores/
   * 
    * Version ChromaVectorStore: 
     const vectorStore = await Chroma.fromDocuments(splits,embeddings , {
      collectionName: "a-test-collection",
      url: "http://localhost:8000",//Par defaut
    }); 
    const retrievedDocs = await vectorStore.similaritySearch(query, 5);
  */
  
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splits,
    embeddings
  );

  const retriever = vectorStore.asRetriever({ k: 6, searchType: "similarity" });
  const retrievedDocs = await retriever.invoke(
    query,
  );
  
  console.log(retrievedDocs);

  return retrievedDocs;
}

// Send a new message to a thread
export async function POST(request, { params: { threadId } }) {
  const { content } = await request.json();

  const context = '';
  const retrievedDocs = await Context(content);

  await openai.beta.threads.messages.create(threadId,{
    role: 'user',
    content: `
            Le CONTEXT BLOCK représente les données mises à jour.
            Ainsi, si l'on vous demande des données et si vos documents et le CONTEXT BLOCK
            n'ont pas les mêmes données sur le même sujet
            vous devez donner la priorité au CONTEXT BLOCK et 
            vous devez préciser que vous avez utilisé les données actualisées
            en précisant qu'ils sont basés sur les données actualisées de la base de données 'COMPAS'.
            Vous utiliserez tout CONTEXT BLOCK
            fournie lors d'une conversation.
            Ne prennnez pas en compte le CONTEXT BLOCK pour 
            proposer des sujets à l'utilisateur quand vous l'accompagnez dans sa recherche.
            START CONTEXT BLOCK
            ${retrievedDocs.map((c: any) => c.pageContent).join("\n")}
            END OF CONTEXT BLOCK`,
  })

  await openai.beta.threads.messages.create(threadId, {
    role: "user",
    content: content,
  });
  const stream = openai.beta.threads.runs.createAndStream(threadId, {
    assistant_id: assistantId,
  });
  return new Response(stream.toReadableStream());
}