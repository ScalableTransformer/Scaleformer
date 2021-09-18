all: paper-scaleformer

paper-scaleformer:
	pdflatex paper-scaleformer.tex
	pdflatex paper-scaleformer.tex
	bibtex paper-scaleformer
	bibtex paper-scaleformer
	pdflatex paper-scaleformer.tex
	pdflatex paper-scaleformer.tex

clean:
	rm -rf *.aux *.bak *.bbl *.blg *.log *.out *.pdf *.sav *.synctex*