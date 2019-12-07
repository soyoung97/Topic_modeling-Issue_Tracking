report:
	pdflatex -shell-escape report.tex
	bibtex report
	pdflatex -shell-escape report.tex
	pdflatex -shell-escape report.tex

clean:
	rm -f report.pdf *.aux report.log report.out report.bbl report.blg
