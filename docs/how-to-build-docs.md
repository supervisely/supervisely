Build documentation guide for debug puproses

**all commands must be executed from the `docs` folder**

- install `doc-requirements.txt`
- then run one of the following commands to build documentation      
  - run `./build_html.sh`
  - run `sphinx-build source/ build/` 
  - `make html`

after building the documentation the `build` folder should appear

if run into some errors or you documentation is not updated try to use:
- `make clean` and then run one of the building commands from above
- also try to refresh page with `shift + f5`
