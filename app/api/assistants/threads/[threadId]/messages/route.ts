import { assistantId } from "@/app/assistant-config";
import { openai } from "@/app/openai";

export const runtime = "nodejs";


// Send a new message to a thread
export async function POST(request, { params: { threadId } }) {
  const { content } = await request.json();

  const rag_context = await fetch('http://127.0.0.1:8000/query/${content}');
  const context = await rag_context.json();


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
            ${context}
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