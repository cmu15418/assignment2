FILES= report.pdf \
	saxpy/ \
	scan/ \
	render/

XFILE = render/snow.par

handin.tar: clean
	tar cvf handin.tar --exclude=$(XFILE) handin.tar $(FILES)
clean:
	(cd saxpy; make clean)
	(cd scan; make clean)
	(cd render; make clean)
	rm -f *~ handin.tar
