FILES := dickens mozilla mr nci ooffice osdb reymont samba sao webster xml x-ray
all: $(FILES) combined

$(FILES):
	wget -O- http://sun.aei.polsl.pl/~sdeor/corpus/$@.bz2 | bzip2 -d > $@

combined: $(FILES)
	cat * >> combined

.PHONY: clean
clean:
	rm -rf $(addsuffix .gz, $(FILES)) $(addsuffix .bz2, $(FILES))
