#! /usr/local/bin/tcsh -f
#
# Script for creating a MTL Distribution

set DESTDIR=/tmp/mtl4.$$


set start=`date`
cat <<EOF

Creating MTL distribution
Started: $start

EOF



echo "*** Copying tree to $DESTDIR..."
set p=`pwd`
rm -rf $DESTDIR
svn export https://svn.osl.iu.edu/tlc/trunk/mtl4/trunk $DESTDIR
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
rm -rf experiment
rm -rf time
rm -f src/sparse_contig2D.h
rm -f src/coordinate.h
rm -f src/MatrixElement.h
rm -f src/MatrixMarket.h
rm -f TODO

# Does not exist in MTL-stable
#
# Move the doc++ stuff up and clean doc/
#
#echo "*** Making postscript and PDF documents"

#cd doc/doc++
#make mtl.pdf mtl.html
#make clean
#cd ../..

#echo "*** Cleaning up doc area"

#rm -rf doc/talks
#rm -rf doc/papers
#rm -rf doc/theses

echo "*** Removing license scripts"

#rm -f insertlic.csh
#rm -f insertlic.sed
#rm -f license.hdr.cpp
#rm -f license.hdr.shell
#rm -f license.hdr.text
rm -f insert_license.py 
rm -f license.short.txt

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

