LATEXMK ?= latexmk
.PHONY: all pdf clean
all: pdf
pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex
clean:
	$(LATEXMK) -C main.tex
