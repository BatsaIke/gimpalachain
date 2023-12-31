import express from 'express';
import { OpenAI } from 'langchain/llms';
import { RetrievalQAChain } from 'langchain/chains';
import { HNSWLib } from 'langchain/vectorstores';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as fs from 'fs';
import cors from 'cors';
import * as dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());
app.use(cors());

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const txtFilename = "data";
const txtPath = path.join(__dirname, `${txtFilename}.txt`);
const VECTOR_STORE_PATH = path.join(__dirname, `${txtFilename}.index`);

// Use your ChatGPT API key
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const model = new OpenAI({ apiKey: OPENAI_API_KEY, model: "text-davinci-002" });

app.get('/', (req, res) => {
  res.send('Hello, Gimpa Assist!');
});

app.post('/ask', async (req, res) => {
  const { question } = req.body;

  if (!question) {
    return res.status(400).json({ error: 'Question is required in the request body.' });
  }

  let vectorStore;

  if (fs.existsSync(VECTOR_STORE_PATH)) {
    console.log('Vector Exists..');
    vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
  } else {
    const text = fs.readFileSync(txtPath, 'utf8').replace(/\r/g, '');
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
    const docs = await textSplitter.createDocuments([text]);
    vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
    await vectorStore.save(VECTOR_STORE_PATH);
  }

  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

  try {
    const result = await chain.call({
      query: question,
    });
    res.json({ response: result });
  } catch (error) {
    console.error(error); 
    res.status(500).json({ error: 'Internal Server Error', message: error.message });
  }
});  

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
