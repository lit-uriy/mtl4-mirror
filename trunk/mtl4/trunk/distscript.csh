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
rm -f boost/numeric/linear_algebra/ets_concepts.hpp
rm -rf boost/numeric/mtl/draft
rm -rf boost/detail
rm -rf boost/property_map
rm -rf boost/sequence
rm -rf libs/numeric/mtl/experimental
rm -rf libs/numeric/mtl/timing
rm -rf libs/property_map
rm -rf libs/sequence

# Also remove deleted directories in SConstruct
grep -v 'libs/numeric/mtl/experimental/' SConstruct | grep -v 'libs/numeric/mtl/timing' > SConstruct.tmp
mv SConstruct.tmp SConstruct


echo "*** Removing license scripts"
rm -f insert_license.py 

exit

echo "*** Making tar"



set TMPDIR=mtl-2.1.2-23
rm -rf $TMPDIR 
mkdir $TMPDIR 
chmod 777 $TMPDIR 
autoconf
autoheader # is autoheader neaded ?
automake
cp -p license.mtl.txt acconfig.h acinclude.m4 aclocal.m4 config.guess config.sub configure configure.in INSTALL VERSION missing install-sh Makefile.am Makefile.in README $TMPDIR
chmod +x $TMPDIR/configure
cp -pR contrib mtl test $TMPDIR
tar czf mtl-2.1.2-23.tar.gz $TMPDIR
cp mtl-*.tar.gz $p

exit

# don't use make dist

#
# ./configure so that we get a Makefile to "make dist"
#
 
echo "*** Configuring (to make distribution tarfile)..."
autoconf
#./configure --prefix=$CONFIGURE_PREFIX $CONFIGURE_ARGS
./configure

rm -f aclocal.m4
aclocal
autoheader
automake --add-missing
chmod +x config.sub config.guess configure install-sh
autoconf
./configure

echo "*** Performing make dist"
REV = `svn info | grep Revision | sed 's/Revision: //'`
# make dist TAR=/usr/local/src/gnu/bin/tar
# make dist TAR=/sw/bin/tar
make dist TAR=`which tar`
mv mtl-*.tar.gz $p
cd ..

#
# All done -- diss the temp area
#
 
echo "*** Removing temporary distribution tree..."
rm -rf mtl.*

cat <<EOF
*** MTL distribution created
 
Started: $start
Ended:   `date`
 
EOF

