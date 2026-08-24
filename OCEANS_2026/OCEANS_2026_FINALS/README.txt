================================================================================
OCEANS 2026 MONTEREY - FINAL SUBMISSION BUNDLE
From Footage to Ethogram: A Deployable Pipeline for Continuous Behavioural
Monitoring of a Captive Octopus
================================================================================

CONTENTS
--------
overleaf_upload.zip              The LaTeX project, ready to upload to Overleaf.
                                 Contains the .tex, IEEEtran.cls and the four
                                 figures, with the .tex at the top level.

octopus_pipeline_oceans2026.tex  Main source. Self-contained: the bibliography
                                 is an inline thebibliography (14 entries), so
                                 there is no .bib file and no bibtex pass.
IEEEtran.cls                     v1.8b, 2015/08/26. Identical to the copy in the
                                 official IEEE conference template (verified;
                                 differs only in line endings). Overleaf also
                                 ships IEEEtran, but this pins the exact version
                                 the reference PDF was built with.
assets/                          The four figures the paper includes. Only these
                                 four are used; the working directory has other
                                 figures that the paper does not reference and
                                 they are deliberately not bundled.

octopus_pipeline_oceans2026.pdf  Reference PDF. Overleaf should reproduce this.
                                 6 pages, 799,210 bytes.

OCEANS_2026_abstract.txt         Plain-text abstract for the web submission form:
                                 title, all seven authors with affiliations and
                                 ORCIDs, corresponding author, abstract in four
                                 paragraphs, keywords. Pure ASCII.
OCEANS_2026_abstract_paste.txt   The abstract alone, for pasting into a single
                                 form field. Pure ASCII, 268 words.

UPLOADING TO OVERLEAF
---------------------
New Project -> Upload Project -> select overleaf_upload.zip
Set the compiler to pdfLaTeX (Menu -> Compiler). No bibtex run is needed.

VERIFIED
--------
Built from these files alone, in a directory containing nothing else, three
pdflatex passes produced a PDF byte-identical to the reference above:

  6 pages, body 5.69 of the 6 allowed, references on page 6
  0 errors, 0 undefined references, 0 undefined citations
  0 missing files, no missing fonts, all fonts embedded
  US Letter (612 x 792 pt)
  9 sections, 4 figures, 5 tables, 14 references, 7 authors
  every figure and table is cited in the text
  no IEEEpubid copyright footer to remove

The one LaTeX warning is an overfull hbox of 1.6 pt, which is not visible.

TEMPLATE COMPLIANCE
-------------------
The IEEE conference template states:

    *CRITICAL: Do Not Use Symbols, Special Characters, Footnotes,
    or Math in Paper Title or Abstract.

Both hold here. The title carries no footnote, and the abstract is plain prose
with no math mode - the shared-first-authorship note is a \thanks inside
\author, which renders as a footnote on page 1 without attaching to the title.
The paper's abstract is a single paragraph, per IEEE style; only the plain-text
copies above are paragraphed, for readability in the submission form.

STILL TO DO - NOT DOABLE FROM THE SOURCE
----------------------------------------
1. IEEE PDF eXpress. OCEANS accepts ONLY PDFs created through it. Everything
   checkable locally passes, but PDF eXpress is the authority. It is gated
   behind completing the IEEE electronic copyright form and registering for the
   conference first.
2. Author order. Shivansh Pachnanda is placed 4th and Harendra Pal Singh 5th by
   inference from the ordinals in the supplied list; those two positions have
   not been confirmed by the authors. Note also that Pachnanda's affiliation is
   given here as "University of Delhi", while the group's companion submission
   places him at the Cluster Innovation Center, as for Singh.
3. Page limit. OCEANS Monterey states 4 to 6 pages plus references. Confirm
   against the final author kit before submitting, since "plus references" and
   "including references" differ by a page here.
