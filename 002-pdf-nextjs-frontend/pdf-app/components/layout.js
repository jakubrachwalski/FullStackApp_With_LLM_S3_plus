import styles from '../styles/layout.module.css'

export default function Layout(props) {
  return (
    <div className={styles.layout}>
      <h1 className={styles.title}>Basic PDF CRUD App</h1>
      {props.children}
    </div>
  )
}