import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.mjs";

let pyodideReadyPromise = loadPyodide();

self.onmessage = async (event) => {
  // make sure loading is done
  const pyodide = await pyodideReadyPromise;
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(["typing_extensions == 4.8"]); // needed for fastapi
  await micropip.install(["ssl"]); // needed for fastapi
  await micropip.install(["fastapi"]); // needed for sdk
  await micropip.install("pyodide-http"); // needed to patch http requests in pyodide

  console.log(
    "Installing Python package:",
    "python-json-logger>=0.1.11, <3.0.0"
  );
  await micropip.install(
    ["python-json-logger>=0.1.11, <3.0.0"],
    (keep_going = true)
  );

  console.log("Installing Python package:", "jsonschema>=2.6.0,<=4.20.0");
  await micropip.install(["jsonschema>=2.6.0,<=4.20.0"], (keep_going = true));

  console.log("Installing Python package:", "aiofiles");
  await micropip.install(["aiofiles"], (keep_going = true));

  console.log("Installing Python package:", "httpx[http2]==0.27.2");
  await micropip.install(["httpx[http2]==0.27.2"], (keep_going = true));

  console.log("Installing Python package:", "requests-toolbelt>=0.9.1");
  await micropip.install(["requests-toolbelt>=0.9.1"], (keep_going = true));

  console.log("Installing Python package:", "tqdm>=4.62.3, <5.0.0");
  await micropip.install(["tqdm>=4.62.3, <5.0.0"], (keep_going = true));

  console.log("Installing Python package:", "python-dotenv>=0.19.2, <=1.0.0");
  await micropip.install(
    ["python-dotenv>=0.19.2, <=1.0.0"],
    (keep_going = true)
  );

  console.log("Installing Python package:", "numpy>=1.19, <2.0.0");
  await micropip.install(["numpy>=1.19, <2.0.0"], (keep_going = true));

  console.log("Installing Python package:", "pyjwt>=2.1.0,<3.0.0");
  await micropip.install(["pyjwt>=2.1.0,<3.0.0"], (keep_going = true));

  console.log("Installing Python package:", "setuptools");
  await micropip.install(["setuptools"], (keep_going = true));

  console.log("Installing Python package:", "pandas>=1.1.3, <=2.1.4");
  await micropip.install(["pandas>=1.1.3, <=2.1.4"], (keep_going = true));

  console.log("Installing Python package:", "pillow>=5.4.1, <=10.2.0");
  await micropip.install(["pillow>=5.4.1, <=10.2.0"], (keep_going = true));

  console.log(
    "Installing Python package:",
    "opencv-python>=4.6.0.66, <5.0.0.0"
  );
  await micropip.install(
    ["opencv-python>=4.6.0.66, <5.0.0.0"],
    (keep_going = true)
  );

  console.log("Installing Python package:", "PyYAML>=5.4.0");
  await micropip.install(["PyYAML>=5.4.0"], (keep_going = true));

  console.log("Installing Python package:", "Shapely>=1.7.1, <=2.0.2");
  await micropip.install(["Shapely>=1.7.1, <=2.0.2"], (keep_going = true));

  console.log("Installing Python package:", "bidict>=0.21.2, <1.0.0");
  await micropip.install(["bidict>=0.21.2, <1.0.0"], (keep_going = true));

  console.log("Installing Python package:", "cachetools>=4.2.3, <=5.5.0");
  await micropip.install(["cachetools>=4.2.3, <=5.5.0"], (keep_going = true));

  console.log("Installing Python package:", "cacheout==0.14.1");
  await micropip.install(["cacheout==0.14.1"], (keep_going = true));

  console.log("Installing Python package:", "protobuf>=3.19.5, <=3.20.3");
  await micropip.install(["protobuf>=3.19.5, <=3.20.3"], (keep_going = true));

  console.log("Installing Python package:", "varname>=0.8.1, <1.0.0");
  await micropip.install(["varname>=0.8.1, <1.0.0"], (keep_going = true));

  console.log("Installing Python package:", "python-magic>=0.4.25, <1.0.0");
  await micropip.install(["python-magic>=0.4.25, <1.0.0"], (keep_going = true));

  console.log("Installing Python package:", "jsonpatch>=1.32, <2.0");
  await micropip.install(["jsonpatch>=1.32, <2.0"], (keep_going = true));

  console.log(
    "Installing Python package:",
    "python-multipart>=0.0.5, <=0.0.12"
  );
  await micropip.install(
    ["python-multipart>=0.0.5, <=0.0.12"],
    (keep_going = true)
  );

  console.log("Installing Python package:", "GitPython");
  await micropip.install(["GitPython"], (keep_going = true));

  console.log("Installing Python package:", "giturlparse");
  await micropip.install(["giturlparse"], (keep_going = true));

  console.log("Installing Python package:", "rich");
  await micropip.install(["rich"], (keep_going = true));

  console.log("Installing Python package:", "click");
  await micropip.install(["click"], (keep_going = true));

  console.log("Installing Python package:", "jinja2>=3.0.3, <4.0.0");
  await micropip.install(["jinja2>=3.0.3, <4.0.0"], (keep_going = true));

  await setupAndImportPythonPackage("./supervisely.tar", "supervisely"); // temporary solution
  await setupAndImportPythonPackage("./src.tar", "src");


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
