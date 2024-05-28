import styles from './compas.module.css'


export default function Compas() {
    return (
        <ul className={styles.example2}>
            <li className={styles.iconContent}>
                <a
                    href="/"
                    aria-label="Home"
                    data-social="home"
                >
                    <div className={styles.filled}></div>
                    <div className={styles.logoContainer}>
                        <img className={styles.logo} src="/logo_compas.png" alt="Compas Logo" />
                    </div>
                </a>
                <div className={styles.tooltip}>Compas</div>
            </li>
        </ul>
    );
}