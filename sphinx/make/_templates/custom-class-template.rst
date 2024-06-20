{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :no-show-inheritance:
   :inherited-members:
   :special-members: __call__

   {% block methods %}
   {% if methods | map('first') | reject('eq', '_') | list | count %}
   .. rubric:: {{ _('Method summary') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes | map('first') | reject('eq', '_') | list | count %}
   .. rubric:: {{ _('Attribute summary') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   .. rubric:: {{ _('Definitions') }}
