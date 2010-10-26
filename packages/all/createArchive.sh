#!/bin/sh

cp ../standard/MTL-*.tar.gz ../examples/MTL-examples*.tar.gz ../tests/MTL-test*.tar.gz . ;
LONGNAME=`ls MTL-all-*.tar.gz | sed 's/\.gz//'` ;
gunzip MTL-*.tar.gz ;
rm MTL-all*.tar.bz2 ;

mv MTL-4*.tar ${LONGNAME} ;
tar -Af ${LONGNAME} MTL-examples-*.tar ;
tar -Af ${LONGNAME} MTL-test*.tar ;

gzip -c ${LONGNAME} > ${LONGNAME}.gz ;
bzip2 ${LONGNAME} ;

rm MTL-examp*.tar ;
rm MTL-test*.tar ;
