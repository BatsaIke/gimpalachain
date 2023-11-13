import express from 'express';
import { OpenAI } from 'langchain/llms';
import { RetrievalQAChain } from 'langchain/chains';
import { HNSWLib } from 'langchain/vectorstores';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as fs from 'fs';
import * as path from 'path';
import * as dotenv from 'dotenv';

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

const txtFilename = "data";

// Use import.meta.url to get the current module's URL
const currentModuleUrl = new URL(import.meta.url);

// Use path.dirname to get the directory path
const moduleDirectory = path.dirname(currentModuleUrl.pathname);

// Combine the directory path with the filename
const txtPath = path.join(moduleDirectory, `${txtFilename}.txt`);
const VECTOR_STORE_PATH = path.join(moduleDirectory, `${txtFilename}.index`);

app.get('/', (req, res) => {
  res.send('Hello, Gimpa Assist!');
});

app.post('/ask', async (req, res) => {
  const { question } = req.body;

  if (!question) {
    return res.status(400).json({ error: 'Question is required in the request body.' });
  }

  const model = new OpenAI({});
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
