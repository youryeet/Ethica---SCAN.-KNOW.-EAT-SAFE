import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.post("/extract-ingredients", async (req, res) => {
  const { text } = req.body;
console.log("Received text length:", text?.length);


  if (!text) return res.status(400).json({ error: "No text provided" });

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-5-mini",
      messages: [
        {
          role: "system",
          content: "You are an assistant that extracts ingredients from text."
        },
        {
          role: "user",
          content: `Extract a clean list of ingredients from this text: ${text}`
        }
      ],
      temperature: 0
    });

    const ingredients = response.choices[0].message.content.trim();
    res.json({ ingredients });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to extract ingredients" });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

