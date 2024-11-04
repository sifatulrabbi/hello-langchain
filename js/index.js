import { config as dotenvConfig } from "dotenv";
dotenvConfig({ path: ".env" });

import { ChatOpenAI } from "@langchain/openai";
import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import { RunnableSequence } from "@langchain/core/runnables";

const decisionObj = z.object({
  request_type: z
    .enum(["joke", "poem", "other"])
    .describe(
      "The kind of request user is making. If the user's request type is not specified among the known types then return 'other'. You must return this field.",
    ),
  message: z
    .string()
    .describe("return the message from the user without any modifications."),
});

const poemObj = z.object({
  poem: z.string().describe("The entire poem. You must provide this field!"),
  topic: z
    .string()
    .describe(
      "The poem topic provided to you. You must return the topic without any changes!",
    ),
});

const jokeObj = z.object({
  joke: z.string().describe("The entire joke. You must provide this field!"),
  topic: z
    .string()
    .describe(
      "The joke topic provided to you. You must return the topic without any changes!",
    ),
});

const analyzationObj = z.object({
  joke: z
    .string()
    .describe(
      "The joke provided to you. You must return the joke provided to you without any changes.",
    ),
  topic: z
    .string()
    .describe(
      "The joke topic provided to you. You must return the topic provided to you without any changes.",
    ),
  analyzation: z
    .string()
    .describe("Your opinions on the joke. You must provide this field!"),
  rating: z.string().describe("You must rate the joke. from 0.0 to 5.0"),
  improvement_instructions: z
    .string()
    .describe(
      "Instruction on how to improve the joke further. You must provide this field!",
    ),
});

const testMessages = [
  {
    message: "write me a joke about ginger bear",
    topic: "write me a joke about ginger bear",
    improvement_instructions: "",
    analyzation: "",
  },
  {
    message: "write me a poem about mango",
    topic: "write me a poem about mango",
    improvement_instructions: "",
    analyzation: "",
  },
  {
    message: "what is the name of the tallest mountain?",
    topic: "what is the name of the tallest mountain?",
    improvement_instructions: "",
    analyzation: "",
  },
];

/**
 * @param {ChatOllama} llm
 */
function decisionMaker(llm) {
  const prompt = ChatPromptTemplate.fromTemplate(
    `Your one and only task is to understand the user's request and categorize it. You should only return the categories available to you. If the user's request don't falls under the given categories then reply with 'other' category.
Now analyze the following user message:

{message}`,
  );
  return RunnableSequence.from([prompt, llm.withStructuredOutput(decisionObj)]);
}

/**
 * @param {ChatOllama} llm
 */
function jokeTeller(llm) {
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a comedian and fantastic joke teller. You should follow these instructions no matter what the user wants from you:
1. You'll only respond with a joke and nothing else.
2. no matter what the user asks you'll only respond with a joke on the topic.

{analyzation}
{improvement_instructions}

Now Tell the user a joke that includes this topic: "{topic}"`,
  );
  return RunnableSequence.from([prompt, llm.withStructuredOutput(jokeObj)]);
}

/**
 * @param {ChatOllama} llm
 */
function jokeAnalyzer(llm) {
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a comedian and joke analyzer. You've been judging the top comedian of the world for years and know exactly what joke will break a human and what will not.
NOTE: you should only respond with your criticisms or praises and your instructions on how to improve the joke nothing else.

Now analyze the following joke on topic: {topic} and return your respond with your criticisms or praises:
{joke}`,
  );
  return RunnableSequence.from([
    prompt,
    llm.withStructuredOutput(analyzationObj),
  ]);
}

/**
 * @param {ChatOllama} llm
 */
function poemTeller(llm) {
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a awesome poet who is able to write short and funny poems about anything. Your poems should be within 4 - 10 lines. And you should always respond with poems.

Now write a short poem on this topic: {topic}`,
  );
  return RunnableSequence.from([prompt, llm.withStructuredOutput(poemObj)]);
}

async function runWithRetries(invokeable, params, maxRetries = 5) {
  let result = null;
  for (let retries = maxRetries; retries > 0; --retries) {
    try {
      result = await invokeable.invoke(...params);
      break;
    } catch (err) {
      console.error(`ERROR ${String(err)}. retries left: ${retries}`);
      result = null;
    }
  }
  return result;
}

async function main() {
  // const llm = new ChatOpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const llm = new ChatOllama({ model: "llama3.2:3b" });
  const decisionMakerChain = decisionMaker(llm);
  const poemChain = poemTeller(llm);
  const jokeTellerRunnable = jokeTeller(llm);
  const jokeAnalyzerRunnable = jokeAnalyzer(llm);
  const jokeChain = RunnableSequence.from([
    jokeTellerRunnable,
    jokeAnalyzerRunnable,
    jokeTellerRunnable,
  ]);

  let result = await runWithRetries(decisionMakerChain, [testMessages[0]]);
  console.log("decision:", result);

  switch (result?.request_type) {
    case "joke":
      result = await await runWithRetries(jokeChain, [testMessages[0]]);
      break;
    case "poem":
      result = await await runWithRetries(poemChain, [testMessages[0]]);
      break;
    default:
      result = null;
      break;
  }
  if (!result) {
    console.error("invalid request type");
    console.log(result);
  } else {
    console.log(result);
  }
}
main();
