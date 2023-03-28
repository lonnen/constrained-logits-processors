=============================
Constrained Logits Processors
=============================

HuggingFace's `transformers library <https://github.com/huggingface/transformers>`_
has a number of logits processors useful for nudging the probability of certain
tokens appearing in text generation models.

This module implements logits processors that constrain logits with an allow
list and a disallow list.

|Build Status|

.. |Build Status| image:: https://github.com/lonnen/constrained-logits-processors/actions/workflows/main.yml/badge.svg?branch=main
   :target: https://github.com/lonnen/constrained-logits-processors/actions/workflows/main.yml

:Code:          https://github.com/lonnen/constrained-logits-processors
:Issues:        https://github.com/lonnen/constrained-logits-processors/issues
:Releases:      https://pypi.org/project/oops_all_itertools/#history
:License:       MIT; See LICENSE

Install
=======

To get started, install the library with `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block:: shell

    $ pip install constrained-logits-processors


Usage
=====

.. code-block:: python

    >>> from constrained_logits_processors import allowListLogitsProcessor, disallowListLogitsProcessor

    >>> # pass
