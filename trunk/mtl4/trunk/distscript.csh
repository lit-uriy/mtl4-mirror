#! /usr/bin/tcsh -f
#
# Script for creating a MTL Distribution

source VERSION

set MTLREPOSITORY='https://svn.osl.iu.edu/tlc/trunk/mtl4/trunk'
set MTLREVISION=`svn info ${MTLREPOSITORY}| grep Revision | sed 's/Revision: //'`

set FULLNAME="mtl${MTLVERSION}-${MTLRELEASE}-r${MTLREVISION}"
echo "Preparing distribution ${FULLNAME}"

set DESTDIR=/tmp/mtl.$$

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
rm -f default.css index.html mtl4.jam Jamroot
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


echo "*** Making tar"
set TARNAME="${FULLNAME}.tar.gz"
tar czf $TARNAME boost Doxyfile INSTALL libs license.short.txt README \
                           SConstruct license.mtl.txt README.scons VERSION
cp $TARNAME $p
cd $p

set MD=`md5sum $TARNAME`
echo "MD5 sum is $MD"

set DOWNLOAD="/l/osl/download/www.osl.iu.edu/research/mtl"

echo "On an OSL machine I can type: cp $TARNAME $DOWNLOAD; chmod a+r ${DOWNLOAD}/${TARNAME}"

#
# All done -- diss the temp area
#
 
echo "*** Removing temporary distribution tree..."
rm -rf $DESTDIR

exit

cp $TARNAME /l/osl/download/www.osl.iu.edu/research/mtl

cat <<EOF
*** MTL distribution created
 
Started: $start
Ended:   `date`
 
EOF

