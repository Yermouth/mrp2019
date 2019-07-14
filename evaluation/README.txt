
CoNLL 2019 Shared Task: Meaning Representation Parsing --- Evaluation Data

Version 1.0; July 1, 2019


Overview
========

This directory contains evaluation data for the MRP 2019 shared task, i.e. the
parser inputs that form the starting point for system submissions.  The files
use a ‘stripped down’ version of the common MRP serialization format, with the
fields to be predicted by participating systems supressed, and some additional
fields indicating which target frameworks can be evaluated for each input.

For general information on the task and the meaning representation frameworks
involved, please see:

  http://mrp.nlpl.eu

The JSON-based uniform interchange format for all frameworks is documented at:

  http://mrp.nlpl.eu/index.php?page=4#format


Contents
========

The main contents in this release is the file providing parser inputs:

  $ wc -l input.mrp
  6288 input.mrp

Here, the number of lines corresponds to the number of parser inputs, i.e. the
evaluation data for MRP 2019 is comprised of 6288 strings to be parsed.  Parser
inputs are presented as empty MRP graphs, where the ‘input’ property provides
the string (each for one sentence-like unit, i.e. one parser input).  Each of
the inputs additionally has a top-level property ‘targets’ that indicates the
range of frameworks that will be evaluated for this sentence.  Because only a
small subset of the MRP 2019 evaluation data is annotated in all frameworks,
most inputs have between one and three elements in their ‘targets’ list.

Additionally, parser inputs are accompanied with the same type of ‘companion’
morpho-syntactic trees as the training data, using the same software version
and parsing model.

  $ wc -l udpipe.mrp
  6288 udpipe.mrp

For additional technical information on the preparation of these companion
analyses, please see the original companion package for the training data:

  http://svn.nlpl.eu/mrp/2019/public/companion.tgz


System Submissions
==================

The files in this archive provide the starting point for participants in the
MRP 2019 shared task.  The evaluation period will run between Monday, July 8,
and Monday, July 22, 2019; no submissions will be possible after that date.

The task emphasizes a cross-framework perspective and invites submissions that
include predicted semantic graphs in all of the five frameworks (DM, PSD, EDS,
UCCA, and AMR).  In this regard, a complete submission will provide graphs for
all parser ‘input’ strings and all ‘targets’ elements, i.e. 13,206 predicted
graphs in total.  All graphs should be concatenated into a single file called
‘output.mrp’ and must be uploaded through the CodaLab submission interface.  In
other words, the outputs of a complete submission should include five separate
graphs for a hypothetical parser input like the following:

  {"id": "20001001", "version": 1.0, "time": "2019-06-23",
   "source": "wsj", "targets": ["dm", "psd", "eds", "ucca", "amr"],
   "input": "Pierre Vinken, 61 years old, will retire soon."}

All submitted graphs must be serialized in the unified MRP interchange format
and must pass the MRP validator.  For background on the file format and tool
support for pre-submission validation, please see:

  http://mrp.nlpl.eu/index.php?page=4#format
  https://github.com/cfmrp/mtool#validation

Further information for participants, including instructions for how to access
the CodaLab site for the task, how to package a submission, the MRP 2019 policy
on re-submissions (within the evaluation deadline), and more, please see:

  http://mrp.nlpl.eu/index.php?page=6


Known Limitations
=================

In general, the MRP task design assumes that parser inputs are ‘raw’ strings,
i.e. follow common conventions regarding punctuation marks and whitespace.  In
the case of some of the AMR ‘input’ values, the strings appear semi-tokenized,
in the sense of separating punctuation marks like commas, periods, quote marks,
and contracted auxiliaries and possessives from adjacent tokens with spurious
whitespace.  Furthermore, some of these strings use (non-standard) conventions
for directional quote marks, viz. the LaTeX-style two-character sequences that
have been popularized in NLP corpora by the Penn Treebank.  For example:

  wb.eng_0003.13  Where 's Homer Simpson when you need him ?
  wb.eng_0003.14  This is a major `` D'oh! '' moment .

For participants starting from the companion morpho-syntactic trees, the first
of these artifacts can have led to wrong quote disambiguation in the tokenizer:
‘straight’ single and double quote marks preceded by whitespace are treated as
left (or opening) quotes, which will at times result in directionally unmatched
quote marks, as well as to contractions whose first character is a left quote
mark rather than an apostrophe.  In retrospect, it turns out that the training
data for MRP 2019 exhibited the same limitations in six of the AMR sub-corpora.
The LaTeX-style quote marks, on the other hand, have been normalized properly
during tokenization for the companion morpho-syntactic trees, i.e. to “ and ”
for the above example.


Release History
===============

[Version 1.0; July 1, 2019]

+ First release of MRP 2019 training data in all frameworks.


Contact
=======

For questions or comments, please do not hesitate to email the task organizers
at: ‘mrp-organizers@nlpl.eu’.

Omri Abend
Jan Hajič
Daniel Hershcovich
Marco Kuhlmann
Stephan Oepen
Tim O'Gorman
Nianwen Xue
