{{ name | escape | underline }}

.. currentmodule:: {{ fullname }}

{% if classes %}
.. rubric:: Classes

.. autosummary::
   :template: autosummary/custom-class-template.rst
    {% for class in classes %}
    {{ class }}
    {% endfor %}

{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
    {% for function in functions %}
    {{ function }}
    {% endfor %}

{% endif %}

.. rubric:: Description

.. automodule:: {{ fullname }}
