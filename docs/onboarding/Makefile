# Makefile for the PNPInverse intern onboarding LaTeX docs.
#
# Targets:
#   make all       -- compile every onboarding PDF
#   make <name>    -- compile a single PDF (e.g. make 01_electrochemistry_primer.pdf)
#   make clean     -- remove LaTeX intermediates
#   make distclean -- also remove generated PDFs

LATEX     := pdflatex
LATEXOPTS := -interaction=nonstopmode -halt-on-error -file-line-error

SOURCES := \
  index.tex \
  00_overview.tex \
  01_electrochemistry_primer.tex \
  02_pnp_bv_model.tex \
  03_numerics_and_codebase.tex \
  04_phase6b_status.tex \
  05_meeting_and_getting_started.tex

PDFS := $(SOURCES:.tex=.pdf)

.PHONY: all clean distclean
all: $(PDFS)

# Two passes so the table of contents picks up.
%.pdf: %.tex preamble.tex
	@echo "==> compiling $<"
	$(LATEX) $(LATEXOPTS) $<
	$(LATEX) $(LATEXOPTS) $<

clean:
	rm -f *.aux *.log *.toc *.out *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(PDFS)
