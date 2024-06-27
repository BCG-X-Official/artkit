.. _faq:

FAQ
===

.. contents::
   :local:
   :depth: 2

About the project
-----------------

What is ARTKIT for?
~~~~~~~~~~~~~~~~~~~

ARTKIT is an Python framework for building fast, flexible, robust
pipelines to automate testing and evaluation of Gen AI systems.

Who developed ARTKIT, and why?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ARTKIT was developed by data scientists and engineers at `BCG X <https://www.bcg.com/x>`_
in response to the growing need for a scalable, efficient, and effective framework
for testing and evaluating Gen AI systems. 

We felt this need acutely in our work with clients, and we wanted to share our solution
with the broader community to help accelerate the development of proficient, equitable, safe,
and secure Gen AI systems.

In the end, we believe all boats rise with the tide of Responsible Gen AI. When Gen AI systems
fail to deliver their intended value, it can cause far reaching harms, erode trust in AI, and
slow down progress for everyone. 

Is ARTKIT an effective substitute for manual testing and evaluation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No! We emphatically believe that manual testing and evaluation is an essential
element of testing any Gen AI system that interacts with humans.
Automated testing and evaluation is a powerful tool for enhancing and supplementing
human efforts, but human expertise, ingenuity, and judgement is
necessary to effectively map risk landscapes, develop strong tests, and
disambiguate tricky evaluations.

About the name
--------------

What is red teaming?
~~~~~~~~~~~~~~~~~~~~

The term *red teaming* originates from the U.S. military, where it
refers to simulating attacks from an adversary to pressure test military
defenses. Later, the term was adopted by the cybersecurity community to
describe simulated attacks on an organization’s IT infrastructure.

In the context of Gen AI systems, we think of red teaming as being
broader than adversarial attacks, because a purely adversarial focus
does not capture the scope of risks presented by this technology. In
particular:

1. Harmful, undesirable behavior of Gen AI systems can arise during both
   adversarial and innocent usage. Thus, it is important to test both
   adversarial and non-adversarial inputs.
2. Guardrails which prevent a Gen AI system from producing certain
   harmful outputs may simultaneously reduce the system’s proficiency in
   its intended domain, exposing a different set of risks. Thus, it is
   important to test for both harmfulness and proficiency.

For these reasons, we often describe any systematic effort to identify
undesirable behavior in a Gen AI system as “red teaming”. However, given
the common association of Red Teaming with adversarial threats, our
official description of ARTKIT includes the phrase *Red Teaming &
Testing* to underscore that our scope is broader than adversarial attacks.

If ARTKIT does Automated Red Teaming and Testing, shouldn’t it be called ARTTKIT?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We acknowledge the inexactitude of our acronym and realize this might trouble our
more exacting users. Our decision reflects a preference for brevity and simplicity 
which we hope facilitates recollection and adoption of our tool.

What does the KIT in ARTKIT stand for?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing. ARTKIT is the Automated Red Teaming (ART) and testing toolkit. We suggested
calling it ARTkit, but marketing said we had to choose between "artkit" and "ARTKIT". 
We thought all-caps would be more readable, so that's how our library became ARTKIT.

Design decisions
----------------

Why do I have to write asynchronous functions to use ARTKIT?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Well, you only *have* to write asynchronous functions to work with ARTKIT's built-in
model classes. ARTKIT pipelines are perfectly capable of running synchronous steps. 
You will just have to create you own model classes if you wish to use ARTKIT to
make synchronous API calls to our supported model providers.
 
This design decision is intended to nudge users towards best practices for
developing pipelines which make large numbers of API calls. While asynchronous
programming may be unfamiliar to some users, it is a relatively simple set of
patterns to learn, and in the age of Gen AI, it is easy to get help writing
asynchronous functions.

Why does ARTKIT use LLMs for everything?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We are biased towards leveraging LLMs throughout the testing and
evaluation process because of the ease of adapting LLMs to the wide
variety and complexity of use cases enabled by Gen AI. While the
compute has a cost, we find that these costs are often dwarfed by the
direct and indirect costs of alternative options:

1. Testing at scale with manual labor typically requires
   recruiting and managing a network of temporary workers
2. Testing at scale with traditional machine learning methods is
   technically challenging to tailor for specific use cases
3. NOT testing at scale leaves an organization exposed to unknown risks

Of course, LLMs are overkill for some tasks and we advocate for using
the simplest approach that gets the job done. ARTKIT components can
easily be customized to use non-LLM techniques. See our tutorial on 
`Creating New Model Classes <user_guide/advanced_tutorials/creating_new_model_classes.ipynb>`_.


If you develop a component which is likely to be useful for a wide range of use
cases, please consider contributing to ARTKIT! Visit our 
:ref:`Contributor Guide <contributor-guide-index>` for more information.


Limitations
-----------


Why can't I push a button and get fully automated report card for my Gen AI system?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unfortunately, when it comes to testing and evaluation, there is no free lunch.
ARTKIT does not provide a "push button" solution because experience has taught us 
that effective testing and evaluation must be tailored to each Gen AI use case.
Automation is a strategy for scaling and accelerating testing and evaluation,
not a substitute for case-specific risk landscape mapping, domain expertise, and critical thinking.

Why can't ARTKIT output a simple metric to tell me if my application is ready to launch?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because it isn't possible, and we emphatically refuse to indulge in dangerous
over-simplifications. Testing and evaluation metrics are influenced by too many
factors and ignore too much context to be meaningful on their own. Readiness for
launch depends on the quality and comprehensiveness of testing and evaluation,
the diversity and severity of persistent failures, the inherent risk of the use
case, and the risk tolerance of stakeholders.

In general, we recommend two criteria for deciding when enough testing has been done:

1. The discovery rate of new system failures has plateaued, indicating
   diminishing returns to further testing. At this point, the remaining
   unresolved failures comprises *residual risk* in the system.
2. The residual risk is reviewed by a committee of stakeholders and
   experts who decide whether the risks are acceptable. The committee
   should also weigh the risks against the expected *benefits* of
   deploying the system.

Using ARTKIT
------------

Can I integrate my Gen AI application into an ARTKIT pipeline?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! Provided your application has an API endpoint, you can create a custom class for
sending and receiving data from your endpoint. See the tutorial on
`Creating New Model Classes <user_guide/advanced_tutorials/creating_new_model_classes.ipynb>`_
in the ARTKIT documentation.

Does ARTKIT support testing and evaluation for multimodal systems?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, ARTKIT supports text-to-image and image-to-text models from
OpenAI (see our tutorial on `Generating and Evaluating Images <user_guide/multimodal/image_generation_and_evaluation.ipynb>`__).

If you need to connect to a model which is not supported by ARTKIT, see our tutorial on
`Creating New Model Classes <user_guide/advanced_tutorials/creating_new_model_classes.ipynb>`_,
and please consider :ref:`Contributor Guide <contributor-guide-index>` to ARTKIT! 

We also welcome contributions of multimodal examples to our `Examples <examples/index.rst>`_
section, which includes end-to-end testing and evaluation examples inspired by real Gen AI use cases.

Contributing
------------

How can I contribute to ARTKIT?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We enthusiastically welcome contributions to ARTKIT!
See the :ref:`Contributor Guide <contributor-guide-index>` for information on
contributing to ARTKIT.

How long will it take for my issue or pull request to get attention?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Response times will vary based on the urgency of the issue, bandwidth of
the maintainers, and complexity of proposed changes. We will make every
effort to respond within 2 weeks, but cannot make guarantees. We ask for
your understanding and emphasize that all contributions are appreciated
even if responses sometimes come slowly.
