{%- extends 'rst.tpl' -%}

{% block input %}
{%- if cell.source.strip() and not cell.source.startswith("%") -%}
.. code:: python

{{ cell.source | indent}}
{% endif -%}
{% endblock input %}

{% block header %}
.. module:: ceedub

.. note:: This tutorial was generated from a Jupyter notebook that can be
          downloaded `here <_static/nb/{{ resources.metadata.name }}.ipynb>`_.

.. _{{resources.metadata.name}}:
{% endblock %}
