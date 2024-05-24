import styles from "./message.module.css"
import logo_compas from "../../../datas/logo_compas.png"
import logo_user from "../../../datas/user.svg"
import Image from "next/image";
import Markdown from "react-markdown";
import JSONAutocomplete from 'json-autocomplete';
import { useEffect } from "react";



type MessageProps = {
  role: "user" | "assistant" | "code";
  text: string | any;
  setNextQuestions:React.Dispatch<React.SetStateAction<string[]>>
  nextQuestions : string[];
};

type AssistantProps = {
  text : any;
  setNextQuestions: any
  nextQuestions : string[]
}

const UserMessage = ({ text }: { text: string }) => {
  return <div className={styles.userContainer}>
            <div className={styles.userIcon}>
                <Image src={logo_user} alt="User Logo" className={styles.icon}/>
            </div>
            <div className={styles.userMessage}>{text}</div>
          </div> 
};

const AnswerMessage = ({ text }: { text: string }) => {
  return (
    <div className={styles.assistantContainer}>
      <div className={styles.assistantIcon}>
        <Image src={logo_compas} alt="Compas Logo" className={styles.icon}/>
      </div>
      <div className={styles.assistantMessage}>
        <Markdown>{text}</Markdown>
      </div>
    </div>  
  );
};

const AssistantMessage = ({ text, setNextQuestions, nextQuestions }: AssistantProps) => {
  console.log('brut: ' + text)
  const jsonText = JSONAutocomplete(text);
  if(jsonText && jsonText.startsWith("{")){//cas non null
    const arrText:string = jsonText.replace('",]}', '",""]}').replace(/\\"/g, '\\n"').replace(/\\n"([\w.-]*)\\n"/g, '\\n\'$1\\n\'').replace(/:\s*}/g, ':""}').replace('{answer','{"answer"').replace('"possible}','","possible":""}').replace(',"possible"}',',"possible":""}').replace('",possible}','","possible":""}').replace('possibleNext}','"possibleNext":""}').replace('""possibleNext":""','","possibleNext":""').replace('possibleNextQuestions}','"possibleNextQuestions":""}').replace(',possibleNextQuestions :',',"possibleNextQuestions" :').replace(',possibleNextQuestions:',',"possibleNextQuestions" :').replace('}undefined','}');
    console.log('arrText: ' + arrText )
    const arrayText:string = arrText.replace(/[_a-zA-Z0-9_.,'\s]+}/g, '}').replace(/:\s*}/g, ':""}')
    console.log('arrayText: ' + arrayText )
    if(arrayText){//cas non null
      const parsedText = JSON.parse(arrayText);
      /* Permet de ne pas boucler sur setNextQuestions */
      useEffect(() => {
        console.log('questions: ' + nextQuestions)
        if (parsedText.possibleNextQuestions) {
          setNextQuestions(parsedText.possibleNextQuestions);
        }
      }, [jsonText]);
      return (<div>
            {parsedText && <AnswerMessage text={parsedText.answer}/>}
          </div>
      );
    }
  }
  return <AnswerMessage text={text}/>;
};

const CodeMessage = ({ text }: { text: string }) => {
  return (
    <div className={styles.codeMessage}>
      {text.split("\n").map((line, index) => (
        <div key={index}>
          <span>{`${index + 1}. `}</span>
          {line}
        </div>
      ))}
    </div>
  );
};

export default function Message ({ role, text, setNextQuestions, nextQuestions }: MessageProps) {
  switch (role) {
    case "user":
      return <UserMessage text={text} />;
    case "assistant":
      return <AssistantMessage text={text} setNextQuestions={setNextQuestions} nextQuestions={nextQuestions}/>;
    case "code":
      return <CodeMessage text={text} />;
    default:
      return null;
  }
};
