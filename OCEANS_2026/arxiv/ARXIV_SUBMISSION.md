# arXiv submission — what to paste where

Upload: **`arxiv_submission.tar.gz`** (LaTeX source, not the PDF — arXiv prefers
source and builds it. Verified: clean-room `pdflatex` x3, 6 pages, 0 errors,
0 undefined references, no Type 3 fonts.)

## Title
From Footage to Ethogram: A Deployable Pipeline for Continuous Behavioural
Monitoring of a Captive Octopus

## Authors (paste in this order)
Siddharth Raj, Krishan Mohan Patel, Himanshu Kumar, Rishabh Thakur,
Shivansh Pachnanda, Harendra Pal Singh, Harald Burgsteiner, Wolfgang Slany

Raj and Patel contributed equally (shared first authorship) — stated in a
footnote on page 1.

## Abstract
Paste the contents of `arxiv_abstract.txt` (plain text, LaTeX stripped).

## Categories
- **Primary: cs.CV** (Computer Vision and Pattern Recognition)
- Cross-list: **q-bio.QM** (Quantitative Methods) — the ethology/behaviour claim
- Cross-list: **cs.LG** — the distillation contribution

## Comments field
6 pages, 4 figures, 5 tables. Accepted at OCEANS 2026 MTS/IEEE Monterey.
Code, labels and frozen benchmarks: https://github.com/sidraj000/octopus-behaviour

## License
Choose **CC BY 4.0** to match the release repo's label licence, or arXiv's
non-exclusive licence if you want to keep redistribution tighter. Do NOT pick
CC BY-NC-SA if you later want IEEE reuse to be frictionless.

# IEEE rules that apply (checked, not assumed)

- **arXiv is IEEE's only approved third-party preprint server.** This is the same
  policy that forced the paper out of the GitHub release repo — arXiv is allowed,
  GitHub is not.
- **Posting before acceptance is permitted.** The footnote reads "Preprint.
  Submitted to OCEANS 2026 MTS/IEEE Monterey." — accurate right now, since the
  paper is submitted and not yet accepted.
- **On acceptance you MUST add the IEEE copyright notice to the arXiv version**
  and replace the footnote's "Submitted to" with "Accepted at". The notice IEEE
  requires is of the form:

      (c) 2026 IEEE. Personal use of this material is permitted. Permission from
      IEEE must be obtained for all other uses, in any current or future media,
      including reprinting/republishing this material for advertising or
      promotional purposes, creating new collective works, for resale or
      redistribution to servers or lists, or reuse of any copyrighted component
      of this work in other works.

  Confirm the exact wording against the IEEE Copyright Form you sign — that form
  is the authority, not this file.
- **On publication**, replace the preprint with either the full IEEE citation
  including DOI, or the accepted version plus the DOI. Not the IEEE-typeset PDF.

# Still to do, after arXiv is live

- **On publication**: replace the preprint with the full IEEE citation including
  the DOI, or the accepted version plus the DOI. Not the IEEE-typeset PDF.
- **Link it from the release repo.** The paper itself cannot be hosted on GitHub
  under IEEE policy, which is why `paper/` was removed, but a link to the arXiv
  preprint is fine and gives the repo the citable reference it currently lacks.

# Note on version parity

This source differs from the OCEANS camera-ready by exactly two footnotes: the
"Accepted at" line and the IEEE copyright notice. Everything else is identical.

**Worth checking in the OCEANS author kit:** some IEEE conferences require the
copyright notice on the bottom of page 1 of the CAMERA-READY too. PDF eXpress
did not flag its absence and it is not in the submitted version, so if the kit
asks for it, the camera-ready needs a rebuild -- the notice is in this arXiv
source and can be lifted straight across.
