all:
	(cd zlib && ./configure)
	$(MAKE) -j -C zlib
	$(MAKE) -C pigz

clean:
	$(MAKE) -C zlib clean
	$(MAKE) -C pigz clean
