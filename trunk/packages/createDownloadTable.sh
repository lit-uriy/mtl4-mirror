rm -rf downloadtable
mkdir -p downloadtable

cp standard/MTL*.tar.gz downloadtable/
FILE_STANDARD=`ls standard/MTL*.tar.gz | sed 's:standard/::'`;

cp examples/MTL*.tar.gz downloadtable/
FILE_EXAMPLES=`ls examples/MTL*.tar.gz | sed 's:examples/::'`;

cp tests/MTL*.tar.gz downloadtable/
FILE_TESTS=`ls tests/MTL*.tar.gz | sed 's:tests/::'`;

MD5_STANDARD=`md5sum standard/MTL*.tar.gz | sed 's/ .*$//'` ;
MD5_EXAMPLES=`md5sum examples/MTL*.tar.gz | sed 's/ .*$//'` ;
MD5_TESTS=`md5sum tests/MTL*.tar.gz | sed 's/ .*$//'`;

sed "s:MTL4_MD5:${MD5_STANDARD}:" downloadtable_base.html > downloadtable/downloadtable.html
sed -i "s:EXAMPLES_MD5:${MD5_EXAMPLES}:" downloadtable/downloadtable.html
sed -i "s:TESTS_MD5:${MD5_TESTS}:" downloadtable/downloadtable.html
sed -i "s:MTL4_FILE:${FILE_STANDARD}:g" downloadtable/downloadtable.html
sed -i "s:EXAMPLES_FILE:${FILE_EXAMPLES}:g" downloadtable/downloadtable.html
sed -i "s:TESTS_FILE:${FILE_TESTS}:g" downloadtable/downloadtable.html
