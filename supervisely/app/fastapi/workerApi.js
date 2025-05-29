// workerApi.mjs
function getPromiseAndResolve() {
  let resolve;
  let promise = new Promise((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

// Each message needs a unique id to identify the response. In a real example,
// we might use a real uuid package
let lastId = 1;
function getId() {
  return lastId++;
}

// Add an id to msg, send it to worker, then wait for a response with the same id.
// When we get such a response, use it to resolve the promise.
function requestResponse(worker, msg) {
  const { promise, resolve } = getPromiseAndResolve();
  const idWorker = getId();
  worker.addEventListener("message", function listener(event) {
    if (event.data?.id !== idWorker) {
      return;
    }
    // This listener is done so remove it.
    worker.removeEventListener("message", listener);
    // Filter the id out of the result
    const { id, ...rest } = event.data;
    resolve(rest);
  });
  worker.postMessage({ id: idWorker, ...msg });
  return promise;
}

const pyodideWorker = new Worker("./supervisely/app/fastapi/webworker.js", { type: "module" });

export function asyncRun(script, context) {
  return requestResponse(pyodideWorker, {
    context,
    python: script,
  });
}
