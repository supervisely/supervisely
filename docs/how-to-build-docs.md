1) Build documentation from docker
  a) check that correct path to the library folder is specified (./repo)
  b) run 'docker-compose up' from 'docs' folder.
  с) in 'docs' should appear 'build' folder with documentation
  d) optional, open in browser 'index.html' from 'build' folder


2) Build documentation locally
  a) check that correct path to the library folder is specified (../../supervisely_lib)
  b) install sphinx and additions which use in documentation:
      pip install sphinx==3.4.3
      pip install sphinx-rtd-theme==0.5.1
      pip install sphinx-copybutton==0.3.1
      pip install m2r2==0.2.7
      pip install nbsphinx==0.8.1
      
  с) run "./build_html.sh" or "sphinx-build source/ build/" from 'docs' folder
  d) in 'docs' should appear 'build' folder with documentation
  e) optional, open in browser 'index.html' from 'build' folder
