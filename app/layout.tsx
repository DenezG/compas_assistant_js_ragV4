import { Rubik } from 'next/font/google'
import "./globals.css";
import Warnings from "./components/warnings";
import { assistantId } from "./assistant-config";
import Background from './background';
import Compas from './compas';
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
        <Compas/>
        {assistantId ? children : <Warnings />}
        <Background />
      </body>
    </html>
  );
}
