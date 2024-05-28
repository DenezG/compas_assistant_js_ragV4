import styles from './loader.module.css'

export default function Loader() {
    return<div className={styles.typewriter}>
            <div className={styles.slide}><i></i></div>
            <div className={styles.paper}></div>
            <div className={styles.keyboard}></div>
        </div>
}