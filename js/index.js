import { config as dotenvConfig } from "dotenv";
import { ChatOpenAI } from "@langchain/openai";
import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";

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

dotenvConfig({ path: ".env" });

/**
 * @param {ChatOllama} llm
 */
async function jokeTeller(llm) {
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a comedian and fantastic joke teller. You should follow these instructions no matter what the user wants from you:
1. You'll only respond with a joke and nothing else.
2. no matter what the user asks you'll only respond with a joke on the topic.

{analyzation}
{improvement_instructions}

Now Tell the user a joke that includes this topic: "{topic}"`,
  );
  return prompt.pipe(llm.withStructuredOutput(jokeObj));
}

/**
 * @param {ChatOllama} llm
 */
async function jokeAnalyzer(llm) {
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a comedian and joke analyzer. You've been judging the top comedian of the world for years and know exactly what joke will break a human and what will not.
NOTE: you should only respond with your criticisms or praises and your instructions on how to improve the joke nothing else.

Now analyze the following joke on topic: {topic} and return your respond with your criticisms or praises:
{joke}`,
  );
  return prompt.pipe(llm.withStructuredOutput(analyzationObj));
}

async function main() {
  // const llm = new ChatOpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const llm = new ChatOllama({ model: "llama3.2:3b" });
  const teller = await jokeTeller(llm);
  const analyzer = await jokeAnalyzer(llm);
  const chain = teller.pipe(analyzer).pipe(teller);

  for (let retries = 5; retries > 0; --retries) {
    try {
      const result = await chain.invoke({
        topic: "ginger bear",
        improvement_instructions: "",
        analyzation: "",
      });
      console.log(result);
      break;
    } catch (err) {
      console.error(`ERROR ${String(err)}. retries left: ${retries}`);
    }
  }
}
main();
