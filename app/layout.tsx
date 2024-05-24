import { Rubik } from 'next/font/google'
import "./globals.css";
import Warnings from "./components/warnings";
import { assistantId } from "./assistant-config";
import Background from './background';
const inter = Rubik({ subsets: ["latin"] });

export const metadata = {
  title: "Compas Assistant",
  description: "A quick chatbot using OpenAI Assistant",
  icons: {
    icon: "/logo_compas.png",
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="logoContainer">
          <a href='/'>
            <img className="logo" src="/logo_compas.png" alt="Compas Logo" />
          </a>
          <a href="https://hublot.lecompas.fr/">
            <img className="logo" src="/logo_hublot.svg" alt="Hublot Logo" />
          </a>
        </div>
        {assistantId ? children : <Warnings />}
        <Background />
      </body>
    </html>
  );
}
