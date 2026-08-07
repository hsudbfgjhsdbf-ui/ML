import '../styles/globals.css';
import Head from 'next/head';

export default function App({ Component, pageProps }) {
  return (
    <>
      <Head>
        <title>Medical Insurance Claim Fraud Detection | IIIT Dharwad</title>
        <meta name="description" content="AI-Driven End-to-End Three-Approach Fraud Detection System (Faculty Adviser: Prof. Ramesh Athe)" />
      </Head>
      <Component {...pageProps} />
    </>
  );
}
