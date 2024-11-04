import { config as dotenvConfig } from "dotenv";
// import { ChatOpenAI } from "@langchain/openai";
import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { z } from "zod";

const jokeObj = z.object({
  joke: z.string().describe("The entire joke."),
});

const analyzationObj = z.object({
  joke: z.string().describe("The joke given to you."),
  analyzation: z.string().describe("Your opinions on the joke."),
  rating: z.string().describe("Rating for the joke. from 0.0 to 5.0"),
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

Now Tell the user a joke that includes this topic: "{topic}"`,
  );
  // prompt = await prompt.partial({
  //    format_instructions: `You'll respond in a valid JSON format. You should follow this format strictly: '{"joke": "Your joke here..."}'`,
  // });
  // /** @type {JsonOutputParser<{joke: string}>} */
  // const parser = new JsonOutputParser();
  return prompt.pipe(llm.withStructuredOutput(jokeObj)); // .pipe(parser);
}

/**
 * @param {ChatOllama} llm
 */
async function jokeAnalyzer(llm) {
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a comedian and joke analyzer. You've been judging the top comedian of the world for years and know exactly what joke will break a human and what will not.
NOTE: you should only respond with your criticisms or praises nothing else.

Now analyze the following joke and return your respond with your criticisms or praises:
<joke>
{joke}
</joke>`,
  );
  // prompt = await prompt.partial({
  //   format_instructions: `You'll respond in a valid JSON format. You should follow this format strictly: '{"analyzation": "Your opinions here..."}'`,
  // });
  // /** @type {JsonOutputParser<{joke: string; analyzation: string; rating: string}>} */
  // const parser = new JsonOutputParser();
  return prompt.pipe(llm.withStructuredOutput(analyzationObj)); // .pipe(parser);
}

async function main() {
  // const gptModel = new ChatOpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const llm = new ChatOllama({ model: "llama3.2:3b", maxRetries: 2 });
  const teller = await jokeTeller(llm);
  const analyzer = await jokeAnalyzer(llm);
  const chain = teller.pipe(analyzer);

  const stream = await chain.stream({ topic: "coding" });
  /** @type {{joke: string; analyzation: string; rating: string}} */
  let result = {};
  for await (const s of stream) {
    console.clear();
    result = s;
    console.log(result);
  }
}
main();
