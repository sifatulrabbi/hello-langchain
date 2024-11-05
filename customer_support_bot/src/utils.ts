import fs from "fs";

const logFilePath = "llmlogs.json";

export async function runWithRetries<T = any>(
  invokeable: { invoke: (...args: any[]) => Promise<T> },
  params: any[],
  maxRetries = 5,
) {
  let result: T | null = null;
  for (let retries = maxRetries; retries > 0; --retries) {
    try {
      result = await invokeable.invoke(...params);
      break;
    } catch (err) {
      console.error(`ERROR ${String(err)}. retries left: ${retries}`);
      // no retry needed since i'm not using local llms anymore.
      process.exit(1);
    }
  }
  return result as T;
}

export function writeLogs(logs: Record<any, any>[]) {
  let content = logs;
  if (fs.existsSync(logFilePath)) {
    const prevContent: any[] = JSON.parse(
      fs.readFileSync(logFilePath).toString("utf-8"),
    );
    content = prevContent.concat(content);
  }
  fs.writeFileSync(logFilePath, JSON.stringify(content, undefined, "  "));
}
