import styles from './formChat.module.css'
import cancelIcon from "../../../datas/cancel.svg"
import Image from "next/image";


interface Props{
    handleSubmit : (e: any) => void
    userInput : string
    setUserInput : (value: React.SetStateAction<string>) => void
    inputDisabled : boolean
    handleCancel : (e: any) => void
}

export default function FormChat(props : Props) {
    return (<div>
        <form
        onSubmit={props.handleSubmit}
        className={`${styles.inputForm} ${styles.clearfix}`}
        >
        <input
          type="text"
          className={styles.input}
          value={props.userInput}
          onChange={(e) => props.setUserInput(e.target.value)}
          placeholder="Entrez votre question"
        />
        {props.inputDisabled? 
          <span onClick={props.handleCancel}
          className={styles.span}
          >
            <Image src={cancelIcon} alt="cancelIcon"/>
            Cancel
          </span>
          :
          <button
          type="submit"
          className={styles.button}
          disabled={props.inputDisabled}
          >
            <div className={styles.svgWrapper}>
              <div className={styles.svgWrapper}>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  width="24"
                  height="24"
                >
                  <path fill="none" d="M0 0h24v24H0z"></path>
                  <path
                    fill="currentColor"
                    d="M1.946 9.315c-.522-.174-.527-.455.01-.634l19.087-6.362c.529-.176.832.12.684.638l-5.454 19.086c-.15.529-.455.547-.679.045L12 14l6-8-8 6-8.054-2.685z"
                  ></path>
                </svg>
                </div>
              </div>
              <span>Envoyer</span>
          </button>
        } 
      </form>
    </div>
    )
}