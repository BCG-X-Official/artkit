{{ fullname | escape | underline }}

.. toctree::
   :maxdepth: 1
   :hidden:

   self
   
.. automodule:: {{ fullname }}
   :no-imported-members:

   {% if "." not in fullname %}
   .. include:: ../api_landing.rst
   {% endif %}

   {% block classes %}
   {% if classes %}
   =======
   Classes
   =======

   .. autosummary::
      :toctree: {{ name }}
      :template: custom-class-template.rst
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   =========
   Functions
   =========

   .. autosummary::
      :toctree: {{ name }}
      :nosignatures:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   =================
   Module attributes
   =================

   .. autosummary::
      :toctree: {{ name }}
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   ==========
   Exceptions
   ==========

   .. autosummary::
      :toctree: {{ name }}
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}

==========
Submodules
==========
.. autosummary::
   :toctree: {{ name }}
   :template: custom-module-template.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
