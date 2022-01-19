1) Build documentation from docker

- check that correct path to the library folder is specified (./repo)
- run 'docker-compose up' from 'docs' folder.
- in 'docs' should appear 'build' folder with documentation
- optional, open in browser 'index.html' from 'build' folder


2) Build documentation locally
- check that correct path to the library folder is specified (../../supervisely_lib)
- install sphinx and additions which use in documentation:
   - pip install sphinx==3.4.3
   - pip install sphinx-material==0.3.5
   - pip install sphinx-copybutton==0.3.1
   - pip install m2r2==0.2.7
   - pip install nbsphinx==0.8.1
      
- run "./build_html.sh" or "sphinx-build source/ build/" from 'docs' folder
- in 'docs' should appear 'build' folder with documentation
- optional, open in browser 'index.html' from 'build' folder

3) venv tbd