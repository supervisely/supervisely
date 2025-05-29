import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.mjs";

let pyodideReadyPromise = loadPyodide();

self.onmessage = async (event) => {
  // make sure loading is done
  const pyodide = await pyodideReadyPromise;
  const { id, python, params } = event.data;
  // Now load any packages we need, run the code, and send the result back.
  await pyodide.loadPackagesFromImports(python);
  // make a Python dictionary with the data from `params`
  const dict = pyodide.globals.get("dict");
//   const globals = dict(Object.entries(params));
  try {
    // Execute the python code in this params

    // const fn = pyodide.runPython(mainScriptTxt);
    //   if (typeof fn === "function") {
    //     const newParams = params.map(p => pyodide.toPy(p));
    //     result = fn(...newParams);
    //   } else {
    //     result = fn;
    //   }


    console.log("Executing Python code with params:", params);
    const fn = await pyodide.runPythonAsync(python);
    if (typeof fn === "function") {
        console.log("Function detected in Python code");
      const newParams = Object.values(params).map((p) => pyodide.toPy(p));
      const result = await fn(...newParams);
      self.postMessage({ result, id });
      return;
    } else {
        console.log("No function detected in Python code, sending as is");
        self.postMessage({ fn, id });
        return;
    }

    // const result = await pyodide.runPythonAsync(python, { globals });
    // self.postMessage({ result, id });
  } catch (error) {
    console.error("Error executing Python code:", error);
    self.postMessage({ error: error.message, id });
  }
};
