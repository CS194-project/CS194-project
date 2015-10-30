all:
	$(MAKE) -j -C zlib
	$(MAKE) -j -C pigz

clean:
	$(MAKE) -C zlib clean
	$(MAKE) -C pigz clean

test:
	$(MAKE) -C zlib test
	$(MAKE) -C pigz test

