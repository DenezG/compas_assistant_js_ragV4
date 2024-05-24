import { assistantId } from "@/app/assistant-config";
import { openai } from "@/app/openai";

export async function POST(request) {
  //console.log("request", request);

  const { fileId } = await request.json();

  //console.log("fileId", fileId);

  const response = await openai.files.content(fileId);

  const image_data = await response.arrayBuffer();
  const image_data_buffer = Buffer.from(image_data);

  //console.log("image_data_buffer", image_data_buffer);

  return new Response(image_data_buffer, {
    headers: {
      "Content-Type": "image/png",
    },
  });
}
