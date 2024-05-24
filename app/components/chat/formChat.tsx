import styles from './formChat.module.css'


interface Props{
    handleSubmit : (e: any) => void
    userInput : string
    setUserInput : (value: React.SetStateAction<string>) => void
    inputDisabled : boolean
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
        <button
          type="submit"
          className={styles.button}
          disabled={props.inputDisabled}
        >
          Envoyer
        </button>
      </form>
    </div>
    )
}