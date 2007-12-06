#! /bin/tcsh -f
#
# Script for creating a MTL Distribution

source VERSION.INPUT

set MTLREPOSITORY='https://svn.osl.iu.edu/tlc/trunk/mtl4/trunk'
set MTLREVISION=`svn info ${MTLREPOSITORY}| grep Revision | sed 's/Revision: //'`

set FULLNAME="mtl${MTLVERSION}-${MTLRELEASE}-r${MTLREVISION}"
echo "Preparing distribution ${FULLNAME}"

echo "*** Creating temporary dir with mtl4 subdir"
set DESTROOTDIR=/tmp/mtl.$$
set DESTDIR=${DESTROOTDIR}/mtl4
mkdir $DESTROOTDIR

set start=`date`
cat <<EOF

Creating MTL distribution
Started: $start

EOF



echo "*** Copying tree to $DESTDIR..."
set p=`pwd`
rm -rf $DESTDIR
svn export ${MTLREPOSITORY} $DESTDIR
cd $DESTDIR



#
# Set social perms
#
echo "*** Setting permissions..."
find . -type d -exec chmod 755 {} \;
find . -type f -exec chmod 644 {} \;
chmod +x insert_license.py


#
# Put in those headers
#
 
echo "*** Inserting license headers..."
umask 022

rm distscript.csh
python insert_license.py license.short.txt cpattern cpppattern scriptpattern README INSTALL

#
# Get rid of stuff not in release
#

echo "*** Removing non-release material"
rm -f default.css index.html mtl4.jam Jamroot VERSION.INPUT
rm -f boost/numeric/linear_algebra/ets_concepts.hpp
rm -rf boost/numeric/mtl/draft
rm -rf boost/detail
rm -rf boost/property_map
rm -rf boost/sequence
rm -rf libs/numeric/mtl/experimental
rm -rf libs/numeric/mtl/timing
rm -rf libs/numeric/mtl/doc/*
rm -rf libs/property_map
rm -rf libs/sequence

# Also remove deleted directories in SConstruct
grep -v 'libs/numeric/mtl/experimental/' SConstruct | grep -v 'libs/numeric/mtl/timing' > SConstruct.tmp
mv SConstruct.tmp SConstruct


echo "*** Removing license scripts"
rm -f insert_license.py 

echo "*** Create version file"
echo "MTLVERSION $MTLVERSION\nMTLRELEASE $MTLRELEASE\nMTLREVISION $MTLREVISION" > VERSION

echo "*** Making tar"
set TARNAME="${FULLNAME}.tar.gz"
#tar czf $TARNAME boost Doxyfile INSTALL libs license.short.txt README \
#                           SConstruct license.mtl.txt README.scons VERSION
cd ..
tar czf $TARNAME mtl4
cp $TARNAME $p

echo "*** Making zip"
set ZIPNAME="${FULLNAME}.zip"
zip -rq $ZIPNAME mtl4
cp $ZIPNAME $p
cd $p

set MDTAR=`md5sum $TARNAME | cut -b 1-32`
set MDZIP=`md5sum $ZIPNAME | cut -b 1-32`

echo "For download file:"
echo "t->addFile("\""Alpha-1 [x]"\"",\n           "\"$TARNAME\"", "\"$MDTAR\"");"
echo "t->addFile("\""Alpha-1 [x]"\"",\n           "\"$ZIPNAME\"", "\"$MDZIP\"");"


set DOWNLOAD="/l/osl/download/www.osl.iu.edu/research/mtl"

echo "On an OSL machine I can type:"
echo "cp $TARNAME $DOWNLOAD"
echo "chmod a+r ${DOWNLOAD}/${TARNAME}"
echo "cp $ZIPNAME $DOWNLOAD"
echo "chmod a+r ${DOWNLOAD}/${ZIPNAME}"

set WEBDIR="/l/osl/www/www.osl.iu.edu/research/mtl/mtl4/doc"
echo "To update the documentations type:"
echo "doxygen"
echo "cp -R libs/numeric/mtl/doc/html/* $WEBDIR"
echo "chmod -R a+rX $WEBDIR"

#
# All done -- diss the temp area
#
 
echo "*** Removing temporary distribution tree..."
rm -rf $DESTROOTDIR

exit

cp $TARNAME /l/osl/download/www.osl.iu.edu/research/mtl

cat <<EOF
*** MTL distribution created
 
Started: $start
Ended:   `date`
 
EOF

