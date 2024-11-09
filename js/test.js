async function theErrFn() {
  await new Promise((r) => setTimeout(r, 200));
  throw new Error("The error");
}

async function childFn() {
  try {
    // return theErrFn(); // is not handled here
    return await theErrFn(); // is handled here
  } catch (err) {
    console.log("from childFn:", err);
  }
}

async function main() {
  try {
    await childFn();
  } catch (err) {
    console.log("from main:", err);
  }
}
main();
