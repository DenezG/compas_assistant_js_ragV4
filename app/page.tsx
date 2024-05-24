"use client";

import React from "react";
import styles from "./page.module.css";
import Background from "./background";

const Home = () => {
  const categories = {
    "Chat simple": "basic-chat",
    "File search": "file-search",
    "Chat avec Images": "all",
  };

  return (
    <main className={styles.main}>
      <div className={styles.title}>
        Chatbot du compas utilisant l'API d'OpenAI
      </div>
      <div className={styles.container}>
        {Object.entries(categories).map(([name, url]) => (
          <a key={name} className={styles.category} href={`/examples/${url}`}>
            {name}
          </a>
        ))}
      </div>
    </main>
  );
};

export default Home;
