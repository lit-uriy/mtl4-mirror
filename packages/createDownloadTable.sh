#!/bin/sh
rm -rf downloadtable
mkdir -p downloadtable

cp standard/MTL-*.tar.gz downloadtable/
cp standard/MTL-*.tar.bz2 downloadtable/
cp standard/MTL-*.rpm downloadtable/
FILE_STANDARD_TGZ=`ls standard/MTL-*.tar.gz | sed 's:standard/::'`;
FILE_STANDARD_TBZ2=`ls standard/MTL-*.tar.bz2 | sed 's:standard/::'`;
FILE_STANDARD_RPM=`ls standard/MTL-*.rpm | sed 's:standard/::'`;

cp examples/MTL-examples*.tar.gz downloadtable/
cp examples/MTL-examples*.tar.bz2 downloadtable/
cp examples/MTL-examples*.rpm downloadtable/
FILE_EXAMPLES_TGZ=`ls examples/MTL-examples*.tar.gz | sed 's:examples/::'`;
FILE_EXAMPLES_TBZ2=`ls examples/MTL-examples*.tar.bz2 | sed 's:examples/::'`;
FILE_EXAMPLES_RPM=`ls examples/MTL-examples*.rpm | sed 's:examples/::'`;

cp tests/MTL-test*.tar.gz downloadtable/
cp tests/MTL-test*.tar.bz2 downloadtable/
cp tests/MTL-test*.rpm downloadtable/
FILE_TESTS_TGZ=`ls tests/MTL-test*.tar.gz | sed 's:tests/::'`;
FILE_TESTS_TBZ2=`ls tests/MTL-test*.tar.bz2 | sed 's:tests/::'`;
FILE_TESTS_RPM=`ls tests/MTL-test*.rpm | sed 's:tests/::'`;

MD5_STANDARD_TGZ=`md5sum standard/MTL-*.tar.gz | sed 's/ .*$//'` ;
MD5_EXAMPLES_TGZ=`md5sum examples/MTL-examples*.tar.gz | sed 's/ .*$//'` ;
MD5_TESTS_TGZ=`md5sum tests/MTL-test*.tar.gz | sed 's/ .*$//'`;

MD5_STANDARD_TBZ2=`md5sum standard/MTL-*.tar.bz2 | sed 's/ .*$//'` ;
MD5_EXAMPLES_TBZ2=`md5sum examples/MTL-examples*.tar.bz2 | sed 's/ .*$//'` ;
MD5_TESTS_TBZ2=`md5sum tests/MTL-test*.tar.bz2 | sed 's/ .*$//'`;

MD5_STANDARD_RPM=`md5sum standard/MTL-*.rpm | sed 's/ .*$//'` ;
MD5_EXAMPLES_RPM=`md5sum examples/MTL-examples*.rpm | sed 's/ .*$//'` ;
MD5_TESTS_RPM=`md5sum tests/MTL-test*.rpm | sed 's/ .*$//'`;

SIZE_STANDARD_TGZ=`ls -sh standard/MTL-*.tar.gz | cut -d' ' -f1` ;
SIZE_EXAMPLES_TGZ=`ls -sh examples/MTL-examples*.tar.gz | cut -d' ' -f1` ;
SIZE_TESTS_TGZ=`ls -sh tests/MTL-tests*.tar.gz | cut -d' ' -f1` ;

SIZE_STANDARD_TBZ2=`ls -sh standard/MTL-*.tar.bz2 | cut -d' ' -f1` ;
SIZE_EXAMPLES_TBZ2=`ls -sh examples/MTL-examples*.tar.bz2 | cut -d' ' -f1` ;
SIZE_TESTS_TBZ2=`ls -sh tests/MTL-tests*.tar.bz2 | cut -d' ' -f1` ;

SIZE_STANDARD_RPM=`ls -sh standard/MTL-*.rpm | cut -d' ' -f1` ;
SIZE_EXAMPLES_RPM=`ls -sh examples/MTL-examples*.rpm | cut -d' ' -f1` ;
SIZE_TESTS_RPM=`ls -sh tests/MTL-tests*.rpm | cut -d' ' -f1` ;

sed "s:MTL4_MD5_TGZ:${MD5_STANDARD_TGZ}:g" downloadtable_base.html > downloadtable/downloadtable.html
sed -i "s:EXAMPLES_MD5_TGZ:${MD5_EXAMPLES_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_MD5_TGZ:${MD5_TESTS_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_FILE_TGZ:${FILE_STANDARD_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_FILE_TGZ:${FILE_EXAMPLES_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_FILE_TGZ:${FILE_TESTS_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_SIZE_TGZ:${SIZE_STANDARD_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_SIZE_TGZ:${SIZE_EXAMPLES_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_SIZE_TGZ:${SIZE_TESTS_TGZ}:g" downloadtable/downloadtable.html

sed -i "s:MTL4_MD5_TBZ2:${MD5_STANDARD_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_MD5_TBZ2:${MD5_EXAMPLES_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_MD5_TBZ2:${MD5_TESTS_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_FILE_TBZ2:${FILE_STANDARD_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_FILE_TBZ2:${FILE_EXAMPLES_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_FILE_TBZ2:${FILE_TESTS_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_SIZE_TBZ2:${SIZE_STANDARD_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_SIZE_TBZ2:${SIZE_EXAMPLES_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_SIZE_TBZ2:${SIZE_TESTS_TBZ2}:g" downloadtable/downloadtable.html

sed -i "s:MTL4_MD5_RPM:${MD5_STANDARD_RPM}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_MD5_RPM:${MD5_EXAMPLES_RPM}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_MD5_RPM:${MD5_TESTS_RPM}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_FILE_RPM:${FILE_STANDARD_RPM}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_FILE_RPM:${FILE_EXAMPLES_RPM}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_FILE_RPM:${FILE_TESTS_RPM}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_SIZE_RPM:${SIZE_STANDARD_RPM}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_SIZE_RPM:${SIZE_EXAMPLES_RPM}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_SIZE_RPM:${SIZE_TESTS_RPM}:g" downloadtable/downloadtable.html
