=============================
Constrained Logits Processors
=============================

HuggingFace's `transformers library <https://github.com/huggingface/transformers>`_
has a number of logits processors useful for nudging the probability of certain
tokens appearing in text generation models.

This module implements a logits processor that constrains logits for text generation
using a letter bank, similar to scrabble, by masking out tokens that cannot be spelled
with the remaining letters to `-math.Inf`. It doesn't work well and maybe it would be 
better implemented as a `Constraint <https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_constraints.py#L5>`.

|Build Status|

.. |Build Status| image:: https://github.com/lonnen/constrained-logits-processors/actions/workflows/main.yml/badge.svg?branch=main
   :target: https://github.com/lonnen/constrained-logits-processors/actions/workflows/main.yml

:Code:          https://github.com/lonnen/constrained-logits-processors
:Issues:        https://github.com/lonnen/constrained-logits-processors/issues
:Releases:      https://pypi.org/project/constrained-logits-processors/#history
:License:       MIT; See LICENSE

Install
=======

To get started, install the library with `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block:: shell

    $ pip install git+https://github.com/lonnen/constrained-logits-processors.git


Usage
=====

.. code-block:: python

    >>> from constrained_logits_processors import LetterBankLogitsProcessor

    >>> # pass
