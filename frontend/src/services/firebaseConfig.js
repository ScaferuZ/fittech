// frontend/src/services/firebaseConfig.js
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

// Your Firebase config - uses environment variables in production, fallback to hardcoded for development
const firebaseConfig = {
  apiKey:
    process.env.REACT_APP_FIREBASE_API_KEY ||
    "AIzaSyBjabp8BVm6-Kv8LMagvWk2bwByiYajHjQ",
  authDomain:
    process.env.REACT_APP_FIREBASE_AUTH_DOMAIN ||
    "fittech-ai-thesis.firebaseapp.com",
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID || "fittech-ai-thesis",
  storageBucket:
    process.env.REACT_APP_FIREBASE_STORAGE_BUCKET ||
    "fittech-ai-thesis.appspot.com",
  messagingSenderId:
    process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID || "772230994937",
  appId:
    process.env.REACT_APP_FIREBASE_APP_ID ||
    "1:772230994937:web:72d9bc88cd943cb114fb28",
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export default app;
