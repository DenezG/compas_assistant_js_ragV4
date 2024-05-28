"use client";

import React, { useState, useEffect, useRef } from "react";
import styles from "./chat.module.css";
import { AssistantStream } from "openai/lib/AssistantStream";
// @ts-expect-error - no types for this yet
import { AssistantStreamEvent } from "openai/resources/beta/assistants/assistants";
import { RequiredActionFunctionToolCall } from "openai/resources/beta/threads/runs/runs";
import Message from "./chat/message";
import Question from "./chat/question";
import FormChat from "./chat/formChat";
import Loader from "./chat/loader";




type ChatProps = {
  functionCallHandler?: (
    toolCall: RequiredActionFunctionToolCall
  ) => Promise<string>;
  setImageUrl ?: any
};

const Chat = ({
  functionCallHandler = () => Promise.resolve(""),setImageUrl,// default to return empty string
}: ChatProps) => {
  const [userInput, setUserInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [inputDisabled, setInputDisabled] = useState(false);
  const [threadId, setThreadId] = useState("");

  // automitcally scroll to bottom of chat
  const messagesEndRef = useRef(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(() => {
    const timer = setTimeout(() => {
      scrollToBottom();
    }, 100); // Adjust the delay time as needed
    
    return () => clearTimeout(timer);
  }, [messages]);

  // create a new threadID when chat component created
  useEffect(() => {
    const createThread = async () => {
      const res = await fetch(`/api/assistants/threads`, {
        method: "POST",
      });
      const data = await res.json();
      setThreadId(data.threadId);
    };
    createThread();
  }, []);

  //Send the message to the api and can cancel the fetch if lag
  const controllerRef = useRef<AbortController>()
  const sendMessage = async (text) => {
    controllerRef.current = new AbortController();
    const signal = controllerRef.current.signal;
    try{
      const response = await fetch(
        `/api/assistants/threads/${threadId}/messages`,
        {
          method: "POST",
          body: JSON.stringify({
            content: text,
          }),
          signal
        }
      )
      const stream = AssistantStream.fromReadableStream(response.body);
      handleReadableStream(stream);
    }catch(e){
      console.log(e);
    }
    
  };

  const submitActionResult = async (runId, toolCallOutputs) => {
    const response = await fetch(
      `/api/assistants/threads/${threadId}/actions`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          runId: runId,
          toolCallOutputs: toolCallOutputs,
        }),
      }
    );
    const stream = AssistantStream.fromReadableStream(response.body);
    handleReadableStream(stream);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!userInput.trim()) return;
    sendMessage(userInput);
    setMessages((prevMessages) => [
      ...prevMessages,
      { role: "user", text: userInput },
    ]);
    setUserInput("");
    setInputDisabled(true);
    scrollToBottom();
  };

  const handleCancel = (e) => {
    e.preventDefault()
    console.log('cancel fun')
    if(controllerRef.current){
      controllerRef.current.abort();
      setInputDisabled(false);
    }
  }

  /* Stream Event Handlers */

  // textCreated - create new assistant message
  const handleTextCreated = () => {
    appendMessage("assistant", "");
  };

  // textDelta - append text to last assistant message
  const handleTextDelta = (delta) => {
    appendToLastMessage(delta.value);
  };

  // toolCallCreated - log new tool call
  const toolCallCreated = (toolCall) => {
    if (toolCall.type != "code_interpreter") return;
    appendMessage("code", "");
  };

  // toolCallDelta - log delta and snapshot for the tool call
  const toolCallDelta = (delta, snapshot) => {
    if (delta.type != "code_interpreter") return;
    if (!delta.code_interpreter.input) return;
    appendToLastMessage(delta.code_interpreter.input);
  };

  // handleRequiresAction - handle function call
  const handleRequiresAction = async (
    event: AssistantStreamEvent.ThreadRunRequiresAction
  ) => {
    const runId = event.data.id;
    const toolCalls = event.data.required_action.submit_tool_outputs.tool_calls;
    // loop over tool calls and call function handler
    const toolCallOutputs = await Promise.all(
      toolCalls.map(async (toolCall) => {
        const result = await functionCallHandler(toolCall);
        return { output: result, tool_call_id: toolCall.id };
      })
    );
    setInputDisabled(true);
    submitActionResult(runId, toolCallOutputs);
  };

  // handleRunCompleted - re-enable the input form
  const handleRunCompleted = () => {
    setInputDisabled(false);
  };

  const handleImageFileDone = async (content: any, snapshot: any) => {
    console.log("handleImageFileDone", content, snapshot);
    const response = await fetch(`/api/assistants/image/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        fileId: content.file_id,
      }),
    }).then((res) => res.blob());
    console.log("response handleImageFileDone", response);
    const url = URL.createObjectURL(response);
    setImageUrl(url);
  };

  const handleReadableStream = (stream: AssistantStream) => {
    // messages
    stream.on("textCreated", handleTextCreated);
    stream.on("textDelta", handleTextDelta);

    // code interpreter
    stream.on("toolCallCreated", toolCallCreated);
    stream.on("toolCallDelta", toolCallDelta);

    // image file
    stream.on("imageFileDone", handleImageFileDone);

    // events without helpers yet (e.g. requires_action and run.done)
    stream.on("event", (event) => {
      if (event.event === "thread.run.requires_action")
        handleRequiresAction(event);
      if (event.event === "thread.run.completed") handleRunCompleted();
    });
  };

  /*
    =======================
    === Utility Helpers ===
    =======================
  */

  const appendToLastMessage = (text) => {
    setMessages((prevMessages) => {
      const lastMessage = prevMessages[prevMessages.length - 1];
      const updatedLastMessage = {
        ...lastMessage,
        text: lastMessage.text + text,
      };
      return [...prevMessages.slice(0, -1), updatedLastMessage];
    });
  };

  const appendMessage = (role, text) => {
    setMessages((prevMessages) => [...prevMessages, { role, text }]);
  };

  /**NextQuestions */
  const [nextQuestions,setNextQuestions] = useState<string[]>([]);;
  return (
    <div className={styles.chatContainer}>
      <div className={styles.messages}>
        {messages.map((msg, index) => (
          <Message key={index} role={msg.role} text={msg.text} setNextQuestions={setNextQuestions} nextQuestions={nextQuestions}/>
        ))}
        {/*Affiche un loader lorsque l'assistant est en train d'écrire/chercher des données*/}
        {inputDisabled && <div  className={styles.loaderContainer}>
                        <Loader/>
                    </div>
        }
        <div ref={messagesEndRef} />
      </div>
       {/* Questions proposées
              * Pour ajouter une question suivre le schéma
              * <Question setInput={props.setInput} status={props.status}
              * question = "Votre question"
              * Le css est à revoir pour plus de 2 questions
        */}
      
              {
                nextQuestions.length>1 ?
                <form onSubmit={handleSubmit} className={styles.questionContainer}> 
                  {nextQuestions.map((question) => (
                    <Question setUserInput={setUserInput} statusOff={inputDisabled} handleSubmit={handleSubmit}
                    question = {question}/>
                  ))}
                </form>
                :
                <form onSubmit={handleSubmit} className={styles.questionContainer}>  
                  <Question setUserInput={setUserInput} statusOff={inputDisabled} handleSubmit={handleSubmit}
                    question = "Bonjour, pouvez-vous m'aider à parcourir les documents téléchargés?"/>
                  <Question setUserInput={setUserInput} statusOff={inputDisabled} handleSubmit={handleSubmit}
                    question = "Pouvez-vous me faire un graphique de l'évolution de la population de mon territoire?"/>
                </form>
              }
              
      
      <FormChat handleSubmit={handleSubmit} userInput={userInput} setUserInput={setUserInput} inputDisabled={inputDisabled} handleCancel={handleCancel}/>
      
    </div>
  );
};

export default Chat;
