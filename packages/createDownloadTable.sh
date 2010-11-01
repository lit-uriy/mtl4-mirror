#!/bin/sh
rm -rf downloadtable
mkdir -p downloadtable

cp standard/MTL-*.tar.gz downloadtable/
cp standard/MTL-*.tar.bz2 downloadtable/
cp standard/MTL-*.rpm downloadtable/
cp standard/MTL-*.zip downloadtable/
FILE_STANDARD_TGZ=`ls standard/MTL-*.tar.gz | sed 's:standard/::'`;
FILE_STANDARD_TBZ2=`ls standard/MTL-*.tar.bz2 | sed 's:standard/::'`;
FILE_STANDARD_RPM=`ls standard/MTL-*.rpm | sed 's:standard/::'`;
FILE_STANDARD_ZIP=`ls standard/MTL-*.zip | sed 's:standard/::'`;

cp examples/MTL-examples*.tar.gz downloadtable/
cp examples/MTL-examples*.tar.bz2 downloadtable/
cp examples/MTL-examples*.rpm downloadtable/
cp examples/MTL-examples*.zip downloadtable/
FILE_EXAMPLES_TGZ=`ls examples/MTL-examples*.tar.gz | sed 's:examples/::'`;
FILE_EXAMPLES_TBZ2=`ls examples/MTL-examples*.tar.bz2 | sed 's:examples/::'`;
FILE_EXAMPLES_RPM=`ls examples/MTL-examples*.rpm | sed 's:examples/::'`;
FILE_EXAMPLES_ZIP=`ls examples/MTL-examples*.zip | sed 's:examples/::'`;

cp tests/MTL-tests*.tar.gz downloadtable/
cp tests/MTL-tests*.tar.bz2 downloadtable/
cp tests/MTL-tests*.rpm downloadtable/
cp tests/MTL-tests*.zip downloadtable/
FILE_TESTS_TGZ=`ls tests/MTL-tests*.tar.gz | sed 's:tests/::'`;
FILE_TESTS_TBZ2=`ls tests/MTL-tests*.tar.bz2 | sed 's:tests/::'`;
FILE_TESTS_RPM=`ls tests/MTL-tests*.rpm | sed 's:tests/::'`;
FILE_TESTS_ZIP=`ls tests/MTL-tests*.zip | sed 's:tests/::'`;

cp all/MTL-all*.tar.gz downloadtable/
cp all/MTL-all*.tar.bz2 downloadtable/
cp all/MTL-all*.rpm downloadtable/
cp all/MTL-all*.zip downloadtable/
FILE_ALL_TGZ=`ls all/MTL-all*.tar.gz | sed 's:all/::'`;
FILE_ALL_TBZ2=`ls all/MTL-all*.tar.bz2 | sed 's:all/::'`;
FILE_ALL_RPM=`ls all/MTL-all*.rpm | sed 's:all/::'`;
FILE_ALL_ZIP=`ls all/MTL-all*.zip | sed 's:all/::'`;

MD5_STANDARD_TGZ=`md5sum standard/MTL-*.tar.gz | sed 's/ .*$//'` ;
MD5_EXAMPLES_TGZ=`md5sum examples/MTL-examples*.tar.gz | sed 's/ .*$//'` ;
MD5_TESTS_TGZ=`md5sum tests/MTL-tests*.tar.gz | sed 's/ .*$//'`;
MD5_ALL_TGZ=`md5sum all/MTL-all*.tar.gz | sed 's/ .*$//'`;

MD5_STANDARD_TBZ2=`md5sum standard/MTL-*.tar.bz2 | sed 's/ .*$//'` ;
MD5_EXAMPLES_TBZ2=`md5sum examples/MTL-examples*.tar.bz2 | sed 's/ .*$//'` ;
MD5_TESTS_TBZ2=`md5sum tests/MTL-tests*.tar.bz2 | sed 's/ .*$//'`;
MD5_ALL_TBZ2=`md5sum all/MTL-all*.tar.bz2 | sed 's/ .*$//'`;

MD5_STANDARD_RPM=`md5sum standard/MTL-*.rpm | sed 's/ .*$//'` ;
MD5_EXAMPLES_RPM=`md5sum examples/MTL-examples*.rpm | sed 's/ .*$//'` ;
MD5_TESTS_RPM=`md5sum tests/MTL-tests*.rpm | sed 's/ .*$//'`;
MD5_ALL_RPM=`md5sum all/MTL-all*.rpm | sed 's/ .*$//'`;

MD5_STANDARD_ZIP=`md5sum standard/MTL-*.zip | sed 's/ .*$//'` ;
MD5_EXAMPLES_ZIP=`md5sum examples/MTL-examples*.zip | sed 's/ .*$//'` ;
MD5_TESTS_ZIP=`md5sum tests/MTL-tests*.zip | sed 's/ .*$//'`;
MD5_ALL_ZIP=`md5sum all/MTL-all*.zip | sed 's/ .*$//'`;

SIZE_STANDARD_TGZ=`ls -sh standard/MTL-*.tar.gz | cut -d' ' -f1` ;
SIZE_EXAMPLES_TGZ=`ls -sh examples/MTL-examples*.tar.gz | cut -d' ' -f1` ;
SIZE_TESTS_TGZ=`ls -sh tests/MTL-tests*.tar.gz | cut -d' ' -f1` ;
SIZE_ALL_TGZ=`ls -sh all/MTL-all*.tar.gz | cut -d' ' -f1` ;

SIZE_STANDARD_TBZ2=`ls -sh standard/MTL-*.tar.bz2 | cut -d' ' -f1` ;
SIZE_EXAMPLES_TBZ2=`ls -sh examples/MTL-examples*.tar.bz2 | cut -d' ' -f1` ;
SIZE_TESTS_TBZ2=`ls -sh tests/MTL-tests*.tar.bz2 | cut -d' ' -f1` ;
SIZE_ALL_TBZ2=`ls -sh all/MTL-all*.tar.bz2 | cut -d' ' -f1` ;

SIZE_STANDARD_RPM=`ls -sh standard/MTL-*.rpm | cut -d' ' -f1` ;
SIZE_EXAMPLES_RPM=`ls -sh examples/MTL-examples*.rpm | cut -d' ' -f1` ;
SIZE_TESTS_RPM=`ls -sh tests/MTL-tests*.rpm | cut -d' ' -f1` ;
SIZE_ALL_RPM=`ls -sh all/MTL-all*.rpm | cut -d' ' -f1` ;

SIZE_STANDARD_ZIP=`ls -sh standard/MTL-*.zip | cut -d' ' -f1` ;
SIZE_EXAMPLES_ZIP=`ls -sh examples/MTL-examples*.zip | cut -d' ' -f1` ;
SIZE_TESTS_ZIP=`ls -sh tests/MTL-tests*.zip | cut -d' ' -f1` ;
SIZE_ALL_ZIP=`ls -sh all/MTL-all*.zip | cut -d' ' -f1` ;

sed "s:MTL4_MD5_TGZ:${MD5_STANDARD_TGZ}:g" downloadtable_base.html > downloadtable/downloadtable.html
sed -i "s:EXAMPLES_MD5_TGZ:${MD5_EXAMPLES_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_MD5_TGZ:${MD5_TESTS_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:ALL_MD5_TGZ:${MD5_ALL_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_FILE_TGZ:${FILE_STANDARD_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_FILE_TGZ:${FILE_EXAMPLES_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_FILE_TGZ:${FILE_TESTS_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:ALL_FILE_TGZ:${FILE_ALL_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_SIZE_TGZ:${SIZE_STANDARD_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_SIZE_TGZ:${SIZE_EXAMPLES_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_SIZE_TGZ:${SIZE_TESTS_TGZ}:g" downloadtable/downloadtable.html
sed -i "s:ALL_SIZE_TGZ:${SIZE_ALL_TGZ}:g" downloadtable/downloadtable.html

sed -i "s:MTL4_MD5_TBZ2:${MD5_STANDARD_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_MD5_TBZ2:${MD5_EXAMPLES_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_MD5_TBZ2:${MD5_TESTS_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:ALL_MD5_TBZ2:${MD5_ALL_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_FILE_TBZ2:${FILE_STANDARD_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_FILE_TBZ2:${FILE_EXAMPLES_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_FILE_TBZ2:${FILE_TESTS_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:ALL_FILE_TBZ2:${FILE_ALL_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_SIZE_TBZ2:${SIZE_STANDARD_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_SIZE_TBZ2:${SIZE_EXAMPLES_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_SIZE_TBZ2:${SIZE_TESTS_TBZ2}:g" downloadtable/downloadtable.html
sed -i "s:ALL_SIZE_TBZ2:${SIZE_ALL_TBZ2}:g" downloadtable/downloadtable.html

sed -i "s:MTL4_MD5_RPM:${MD5_STANDARD_RPM}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_MD5_RPM:${MD5_EXAMPLES_RPM}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_MD5_RPM:${MD5_TESTS_RPM}:g" downloadtable/downloadtable.html
sed -i "s:ALL_MD5_RPM:${MD5_ALL_RPM}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_FILE_RPM:${FILE_STANDARD_RPM}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_FILE_RPM:${FILE_EXAMPLES_RPM}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_FILE_RPM:${FILE_TESTS_RPM}:g" downloadtable/downloadtable.html
sed -i "s:ALL_FILE_RPM:${FILE_ALL_RPM}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_SIZE_RPM:${SIZE_STANDARD_RPM}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_SIZE_RPM:${SIZE_EXAMPLES_RPM}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_SIZE_RPM:${SIZE_TESTS_RPM}:g" downloadtable/downloadtable.html
sed -i "s:ALL_SIZE_RPM:${SIZE_ALL_RPM}:g" downloadtable/downloadtable.html

sed -i "s:MTL4_MD5_ZIP:${MD5_STANDARD_ZIP}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_MD5_ZIP:${MD5_EXAMPLES_ZIP}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_MD5_ZIP:${MD5_TESTS_ZIP}:g" downloadtable/downloadtable.html
sed -i "s:ALL_MD5_ZIP:${MD5_ALL_ZIP}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_FILE_ZIP:${FILE_STANDARD_ZIP}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_FILE_ZIP:${FILE_EXAMPLES_ZIP}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_FILE_ZIP:${FILE_TESTS_ZIP}:g" downloadtable/downloadtable.html
sed -i "s:ALL_FILE_ZIP:${FILE_ALL_ZIP}:g" downloadtable/downloadtable.html
sed -i "s:MTL4_SIZE_ZIP:${SIZE_STANDARD_ZIP}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_SIZE_ZIP:${SIZE_EXAMPLES_ZIP}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_SIZE_ZIP:${SIZE_TESTS_ZIP}:g" downloadtable/downloadtable.html
sed -i "s:ALL_SIZE_ZIP:${SIZE_ALL_ZIP}:g" downloadtable/downloadtable.html
