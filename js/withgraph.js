import { config as dotenvConfig } from "dotenv";
dotenvConfig({ path: ".env" });

import { ChatOpenAI } from "@langchain/openai";
import { ChatOllama } from "@langchain/ollama";
import { tool } from "@langchain/core/tools";
import { HumanMessage } from "@langchain/core/messages";
import { Annotation, MemorySaver, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { z } from "zod";

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

const StateAnnotation = Annotation.Root({
  messages: Annotation({
    reducer: (x, y) => {
      console.log("Annotation.Root.messages.reducer, x:", x, "\ny:", y);
      return x.concat(y);
    },
    default: () => [],
  }),
});

const weatherTool = tool(
  async ({ query }) => {
    if (
      query.toLowerCase().includes("sf") ||
      query.toLowerCase().includes("san francisco")
    ) {
      return "It's 60 degrees and foggy.";
    }
    return "It's 90 degrees and sunny.";
  },
  {
    name: "get_weather_info",
    description:
      "Call to get the current weather for a location or a city or a country.",
    schema: z.object({
      query: z.string().describe("The query to use in your search."),
    }),
  },
);

const tools = [weatherTool];
const toolNode = new ToolNode(tools);

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  apiKey: process.env.OPENAI_API_KEY,
}).bindTools(tools);
// const llm = new ChatOllama({ model: "llama3.2:3b" }).bindTools(tools);

/** @param {StateAnnotation.state} state */
function shouldContinue({ messages }) {
  console.log("shouldContinue:", messages);
  const lastMessage = messages.at(-1);
  if (lastMessage?.tool_calls?.length) {
    return "tools";
  }
  return "__end__";
}

/** @param {StateAnnotation.state} state */
async function callModel({ messages }) {
  const response = await llm.invoke(messages);
  return { messages: [response] };
}

const workflow = new StateGraph(StateAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tools", "agent");

const checkpointer = new MemorySaver();
const runnable = workflow.compile({ checkpointer });

async function main() {
  const finalState = await runWithRetries(runnable, [
    { messages: [new HumanMessage("what is the weather in san francisco")] },
    { configurable: { thread_id: "abcdefgh" } },
  ]);
  console.log(finalState.messages[finalState.messages.length - 1].content);

  const nextState = await runWithRetries(runnable, [
    { messages: [new HumanMessage("what about ny")] },
    { configurable: { thread_id: "abcdefgh" } },
  ]);
  console.log(nextState.messages[nextState.messages.length - 1].content);
}
main();
