import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";
import vision from "@google-cloud/vision";
import fetch from "node-fetch";

dotenv.config();
const app = express();

// ----------------- CORS CONFIG -----------------
const corsOptions = {
  origin: "http://127.0.0.1:5501", // your frontend URL
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type"],
  credentials: true,
};

app.use(cors(corsOptions)); // MUST be before your routes


// ----------------- JSON PARSING -----------------
app.use(express.json({ limit: "10mb" }));

// ----------------- TEST CORS ENDPOINT -----------------
app.get("/test-cors", (req, res) => {
  res.json({ message: "✅ CORS is working properly" });
});

// ----------------- START SERVER -----------------
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
