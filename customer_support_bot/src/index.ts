import { config as dotenvConfig } from "dotenv";
dotenvConfig({ path: ".env" });

// import { ChatOllama } from "@langchain/ollama";
import { ChatOpenAI } from "@langchain/openai";
import {
  Annotation,
  MessagesAnnotation,
  NodeInterrupt,
  StateGraph,
  MemorySaver,
} from "@langchain/langgraph";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { writeLogs } from "./utils";

// const llm = new ChatOllama({
//   model: "llama3.2:3b",
//   temperature: 0,
// });
const llm = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-4o-mini",
  temperature: 0,
});

const StateAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  nextRepresentative: Annotation<string>,
  refundAuthorized: Annotation<string>,
});

async function initialSupport(state: typeof StateAnnotation.State) {
  const SYSTEM_TEMPLATE = `You are frontline support staff for LangCorp, a company that sells computers.
Be concise in your responses.
You can chat with customers and help them with basic questions, but if the customer is having a billing or technical problem,
do not try to answer the question directly or gather information.
Instead, immediately transfer them to the billing or technical team by asking the user to hold for a moment.
Otherwise, just respond conversationally.`;
  const supportResp = await llm.invoke([
    { role: "system", content: SYSTEM_TEMPLATE },
    ...state.messages,
  ]);

  const CATEGORIZATION_SYSTEM_TEMPLATE = `You are an expert customer support routing system.
Your job is to detect whether a customer support representative is routing a user to a billing team or a technical team, or if they are just responding conversationally.`;
  const CATEGORIZATION_HUMAN_TEMPLATE = `The previous conversation is an interaction between a customer support representative and a user.
Extract whether the representative is routing the user to a billing or technical team, or whether they are just responding conversationally.
Respond with a JSON object containing a single key called "nextRepresentative" with one of the following values:

If they want to route the user to the billing team, respond only with the word "BILLING".
If they want to route the user to the technical team, respond only with the word "TECHNICAL".
Otherwise, respond only with the word "RESPOND".`;

  const schema = z.object({
    nextRepresentative: z.enum(["BILLING", "TECHNICAL", "RESPOND"]),
  });
  const schemaDescriptor = zodToJsonSchema(schema);
  const categorizationResp = await llm.invoke(
    [
      { role: "system", content: CATEGORIZATION_SYSTEM_TEMPLATE },
      ...state.messages,
      { role: "user", content: CATEGORIZATION_HUMAN_TEMPLATE },
    ],
    {
      response_format: {
        type: "json_schema",
        json_schema: {
          schema,
          name: "categorization_response_schema",
          description: schemaDescriptor.description,
          strict: true,
        },
      },
    },
  );
  const categorizationOutput = JSON.parse(categorizationResp.content as string);

  return {
    messages: supportResp,
    nextRepresentative: categorizationOutput.nextRepresentative,
  };
}

async function billingSupport(state: typeof StateAnnotation.State) {
  const SYSTEM_TEMPLATE = `You are an expert billing support specialist for LangCorp, a company that sells computers.
Help the user to the best of your ability, but be concise in your responses.
You have the ability to authorize refunds, which you can do by transferring the user to another agent who will collect the required information.
If you do, assume the other agent has all necessary information about the customer and their order.
You do not need to ask the user for more information.

Help the user to the best of your ability, but be concise in your responses.`;

  let trimmedHistory = state.messages;
  if (trimmedHistory.at(-1)?.getType() === "ai") {
    trimmedHistory.slice(0, -1);
  }

  const billingSupportResp = await llm.invoke([
    { role: "system", content: SYSTEM_TEMPLATE },
    ...trimmedHistory,
  ]);

  const CATEGORIZATION_SYSTEM_TEMPLATE = `Your job is to detect whether a billing support representative wants to refund the user.`;
  const CATEGORIZATION_HUMAN_TEMPLATE = `The following text is a response from a customer support representative.
Extract whether they want to refund the user or not.
Respond with a JSON object containing a single key called "nextRepresentative" with one of the following values:

If they want to refund the user, respond only with the word "REFUND".
Otherwise, respond only with the word "RESPOND".

Here is the text:

<text>
${billingSupportResp.content as string}
</text>.`;
  const schema = z.object({
    nextRepresentative: z.enum(["REFUND", "RESPOND"]),
  });
  const schemaDescriptor = zodToJsonSchema(
    z.object({
      nextRepresentative: z.enum(["REFUND", "RESPOND"]),
    }),
  );
  const categorizationResp = await llm.invoke(
    [
      { role: "system", content: CATEGORIZATION_SYSTEM_TEMPLATE },
      { role: "user", content: CATEGORIZATION_HUMAN_TEMPLATE },
    ],
    {
      response_format: {
        type: "json_schema",
        json_schema: {
          schema,
          name: "categorization_response",
          description: schemaDescriptor.description,
          strict: true,
        },
      },
    },
  );
  const categorizationOutput = JSON.parse(categorizationResp.content as string);

  return {
    messages: billingSupportResp,
    nextRepresentative: categorizationOutput.nextRepresentative,
  };
}

async function technicalSupport(state: typeof StateAnnotation.State) {
  const SYSTEM_TEMPLATE = `You are an expert at diagnosing technical computer issues. You work for a company called LangCorp that sells computers.
Help the user to the best of your ability, but be concise in your responses.`;

  let trimmedHistory = state.messages;
  if (trimmedHistory.at(-1)?.getType() === "ai") {
    trimmedHistory = trimmedHistory.slice(0, -1);
  }
  const resp = await llm.invoke([
    { role: "system", content: SYSTEM_TEMPLATE },
    ...trimmedHistory,
  ]);

  return {
    messages: resp,
  };
}

async function handleRefund(state: typeof StateAnnotation.State) {
  if (!state.refundAuthorized) {
    console.log("--- HUMAN AUTHORIZATION REQUIRED FOR REFUND ---");
    throw new NodeInterrupt("Human authorization required.");
  }
  return {
    messages: {
      role: "assistant",
      content: "Refund processed!",
    },
  };
}

let builder = new StateGraph(StateAnnotation)
  .addNode("initial_support", initialSupport)
  .addNode("billing_support", billingSupport)
  .addNode("technical_support", technicalSupport)
  .addNode("handle_refund", handleRefund)
  .addEdge("__start__", "initial_support")
  .addConditionalEdges("initial_support", async (state) => {
    if (state.nextRepresentative.includes("BILLING")) {
      return "billing_support";
    } else if (state.nextRepresentative.includes("TECHNICAL")) {
      return "technical_support";
    } else {
      return "__end__";
    }
  })
  .addEdge("technical_support", "__end__")
  .addConditionalEdges("billing_support", async (state) => {
    if (state.nextRepresentative.includes("REFUND")) {
      return "handle_refund";
    } else {
      return "__end__";
    }
  })
  .addEdge("handle_refund", "__end__");

const checkpointer = new MemorySaver();
const graph = builder.compile({ checkpointer });

/* // code for ts lab only
import * as tslab from "tslab";

const representation = graph.getGraph();
const image = await representation.drawMermaidPng();
const arrayBuffer = await image.arrayBuffer();

await tslab.display.png(new Uint8Array(arrayBuffer));
*/

async function main() {
  const stream = await graph.stream(
    {
      messages: [
        {
          role: "user",
          content:
            "I've changed my mind and I want a refund for order #182818!",
        },
      ],
    },
    {
      configurable: {
        thread_id: "refund_testing_id",
      },
    },
  );
  for await (const value of stream) {
    writeLogs([value]);
  }

  const currState = await graph.getState({
    configurable: { thread_id: "refund_testing_id" },
  });
  console.log(
    "CURRENT TASKS",
    JSON.stringify(currState.tasks, null, 2),
    "\n\n",
  );
  console.log("NEXT TASKS", currState.next, "\n\n");

  await graph.updateState(
    { configurable: { thread_id: "refund_testing_id" } },
    { refundAuthorized: true },
  );

  const resumedStream = await graph.stream(null, {
    configurable: { thread_id: "refund_testing_id" },
  });
  for await (const value of resumedStream) {
    console.log(value);
  }

  const technicalStream = await graph.stream(
    {
      messages: [
        {
          role: "user",
          content:
            "My LangCorp computer isn't turning on because I dropped it in water.",
        },
      ],
    },
    {
      configurable: {
        thread_id: "technical_testing_id",
      },
    },
  );
  for await (const value of technicalStream) {
    console.log(value);
  }

  const conversationalStream = await graph.stream(
    {
      messages: [
        {
          role: "user",
          content: "How are you? I'm Cobb.",
        },
      ],
    },
    {
      configurable: {
        thread_id: "conversational_testing_id",
      },
    },
  );
  for await (const value of conversationalStream) {
    console.log(value);
  }
}
main();
